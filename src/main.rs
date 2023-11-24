#![feature(portable_simd)]
use std::array;
use std::simd::Simd;

const N: usize = 1024;
const BLOCK_SIZE: usize = 1;

fn multiply_vector_size4(v1: [f32; 4], v2: [f32; 4]) -> core::arch::aarch64::float32x4_t {
    let sv1: Simd<f32, 4> = Simd::from_array(v1);
    let sv2: Simd<f32, 4> = Simd::from_array(v2);

    let archv1 = core::arch::aarch64::float32x4_t::from(sv1);
    let archv2 = core::arch::aarch64::float32x4_t::from(sv2);

    unsafe { core::arch::aarch64::vmulq_f32(archv1, archv2) }
}

fn measure_time<F>(mut f: F) -> f32
where
    F: FnMut() -> (),
{
    let start_time = std::time::Instant::now();
    f();
    let end_time = std::time::Instant::now();
    let duration = end_time.duration_since(start_time);
    duration.as_secs_f32()
}

fn matmul_gemm(a: &Vec<Vec<f32>>, b: &Vec<Vec<f32>>, c: &mut Vec<Vec<f32>>) {
    for i in 0..N {
        for j in 0..N {
            for k in 0..N {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
}

fn matmul_gemm_local_accumulator(a: &Vec<Vec<f32>>, b: &Vec<Vec<f32>>, c: &mut Vec<Vec<f32>>) {
    for i in 0..N {
        for j in 0..N {
            let mut acc = 0.0;
            for k in 0..N {
                acc += a[i][k] * b[k][j];
            }
            c[i][j] += acc;
        }
    }
}

fn matmul_gemm_local_transposed(a: &Vec<Vec<f32>>, b: &Vec<Vec<f32>>, c: &mut Vec<Vec<f32>>) {
    for i in 0..N {
        for j in 0..N {
            let mut acc = 0.0;
            for k in 0..N {
                acc += a[i][k] * b[j][k];
            }
            c[i][j] += acc;
        }
    }
}

fn matmul_gemm_block(a: &Vec<Vec<f32>>, b: &Vec<Vec<f32>>, c: &mut Vec<Vec<f32>>) {
    let block_num = N / BLOCK_SIZE;
    for bi in 0..block_num {
        for bj in 0..block_num {
            for bk in 0..block_num {
                // Block GEMM
                for i in 0..BLOCK_SIZE {
                    for j in 0..BLOCK_SIZE {
                        let mut acc = 0.0;
                        for k in 0..BLOCK_SIZE {
                            acc += a[bi * BLOCK_SIZE + i][bk * BLOCK_SIZE + k]
                                * b[bj * BLOCK_SIZE + k][bk * BLOCK_SIZE + j];
                        }
                        c[bi * BLOCK_SIZE + i][bj * BLOCK_SIZE + j] += acc;
                    }
                }
            }
        }
    }
}

fn matmul_gemm_simple_neon(a: &Vec<Vec<f32>>, b: &Vec<Vec<f32>>, c: &mut Vec<Vec<f32>>) {
    // assume b is transposed
    for i in 0..N {
        for j in 0..N {
            //
            let mut acc = 0.0;
            for chunk in (0..N).step_by(4) {
                let a_chunk: [f32; 4] = [
                    a[i][chunk],
                    a[i][chunk + 1],
                    a[i][chunk + 2],
                    a[i][chunk + 3],
                ];
                let b_chunk: [f32; 4] = [
                    b[j][chunk],
                    b[j][chunk + 1],
                    b[j][chunk + 2],
                    b[j][chunk + 3],
                ];

                let vres = multiply_vector_size4(a_chunk, b_chunk);
                // https://doc.rust-lang.org/core/arch/aarch64/fn.vaddvq_f32.html
                let res = unsafe { core::arch::aarch64::vaddvq_f32(vres) };
                acc += res;
            }
            c[i][j] = acc;
        }
    }
}

fn main() {
    // create a 2d array of size N x N
    // use the vec! macro to initialize the array

    let mut a: Vec<Vec<f32>> = vec![vec![0.0; N]; N];
    let mut b: Vec<Vec<f32>> = vec![vec![0.0; N]; N];
    let mut c: Vec<Vec<f32>> = vec![vec![0.0; N]; N];
    let mut truth: Vec<Vec<f32>> = vec![vec![0.0; N]; N];

    // load a and b from A.txt and B.txt
    let a_content = std::fs::read_to_string("data/A.txt").unwrap();
    let b_content = std::fs::read_to_string("data/B.txt").unwrap();
    let truth_content = std::fs::read_to_string("data/C.txt").unwrap();

    for (i, line) in a_content.lines().enumerate() {
        for (j, num) in line.split_whitespace().enumerate() {
            a[i][j] = num.parse::<f32>().unwrap();
        }
    }

    for (i, line) in b_content.lines().enumerate() {
        for (j, num) in line.split_whitespace().enumerate() {
            b[i][j] = num.parse::<f32>().unwrap();
        }
    }

    for (i, line) in truth_content.lines().enumerate() {
        for (j, num) in line.split_whitespace().enumerate() {
            truth[i][j] = num.parse::<f32>().unwrap();
        }
    }

    // some algos require b to be transposed
    let mut b_transposed: Vec<Vec<f32>> = vec![vec![0.0; N]; N];
    for i in 0..N {
        for j in 0..N {
            b_transposed[i][j] = b[j][i];
        }
    }

    let flop = 2 * N * N * N;

    // naive gemm
    {
        let name = "matmul_gemm";
        let duration = measure_time(|| matmul_gemm(&a, &b, &mut c));
        print!(
            "{:<30}: {:>5} GFLOP/s\n",
            name,
            flop as f32 / duration / 1e9
        );

        // make sure C is same as the truth
        check(&c, &truth);
        c = vec![vec![0.0; N]; N];
    }

    // gemm local accumulator
    {
        let name = "matmul_gemm_local_accumulator";
        // ...

        let duration = measure_time(|| matmul_gemm_local_accumulator(&a, &b, &mut c));
        print!(
            "{:<30}: {:>5} GFLOP/s\n",
            name,
            flop as f32 / duration / 1e9
        );
        check(&c, &truth);
        c = vec![vec![0.0; N]; N];
    }

    // gemm local transposed
    {
        let name = "matmul_gemm_local_transposed";
        let duration = measure_time(|| matmul_gemm_local_transposed(&a, &b_transposed, &mut c));
        print!(
            "{:<30}: {:>5} GFLOP/s\n",
            name,
            flop as f32 / duration / 1e9
        );
        check(&c, &truth);
        c = vec![vec![0.0; N]; N];
    }

    // gemm simple neon
    {
        let name = "matmul_gemm_simple_neon";
        let duration = measure_time(|| matmul_gemm_simple_neon(&a, &b_transposed, &mut c));
        print!(
            "{:<30}: {:>5} GFLOP/s\n",
            name,
            flop as f32 / duration / 1e9
        );
        check(&c, &truth);
        c = vec![vec![0.0; N]; N];
    }

    // gemm with blocks
    {
        let name = "matmul_gemm_block";
        let duration = measure_time(|| matmul_gemm_block(&a, &b_transposed, &mut c));
        print!(
            "{:<30}: {:>5} GFLOP/s\n",
            name,
            flop as f32 / duration / 1e9
        );
        check(&c, &truth);
        // c = vec![vec![0.0; N]; N];
    }
}

fn check(c: &Vec<Vec<f32>>, truth: &Vec<Vec<f32>>) {
    for i in 0..N {
        for j in 0..N {
            assert!(
                (c[i][j] - truth[i][j]).abs() < 1e-2,
                "c[{}][{}] = {} != {}",
                i,
                j,
                c[i][j],
                truth[i][j]
            );
        }
    }
}
