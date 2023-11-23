#![feature(portable_simd)]

use std::simd::Simd;

const N: usize = 2048;

fn multiply_vector_size4(v1: [f32; 4], v2: [f32; 4]) -> core::arch::aarch64::float32x4_t {
    let sv1: Simd<f32, 4> = Simd::from_array(v1);
    let sv2: Simd<f32, 4> = Simd::from_array(v2);

    let archv1 = core::arch::aarch64::float32x4_t::from(sv1);
    let archv2 = core::arch::aarch64::float32x4_t::from(sv2);

    unsafe { core::arch::aarch64::vmulq_f32(archv1, archv2) }
}

fn matmul(a: &Vec<Vec<f32>>, b: &Vec<Vec<f32>>, c: &mut Vec<Vec<f32>>) {
    for i in 0..N {
        for j in 0..N {
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

    // preprocessing
    // transpose the matrix b
    for i in 0..N {
        for j in 0..i {
            let tmp = b[i][j];
            b[i][j] = b[j][i];
            b[j][i] = tmp;
        }
    }

    let flop = 2 * N * N * N;

    let start_time = std::time::Instant::now();

    matmul(&a, &b, &mut c);

    let end_time = std::time::Instant::now();
    let duration = end_time.duration_since(start_time);

    // print the flops
    print!("{} GFLOP/s\n", flop as f32 / duration.as_secs_f32() / 1e9);

    // make sure C is same as the truth
    for i in 0..N {
        for j in 0..N {
            assert!((c[i][j] - truth[i][j]).abs() < 1e-3);
        }
    }
}
