use std::array;

const N: usize = 1024;

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

    // matrix multiplication
    for i in 0..N {
        for j in 0..N {
            // lets use simd to speed up the computation

            let res: [f32; N] = array::from_fn(|k| a[i][k] * b[j][k]);
            c[i][j] = res.iter().sum();

            // for k in 0..N {
            //     c[i][j] += a[i][k] * b[j][k];
            // }
        }
    }

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
