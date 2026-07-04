// First implementation of sumcheck protocol using arkworks

mod improved;
mod utils;

use ark_ff::Field;
use ark_linear_sumcheck::ml_sumcheck::MLSumcheck;
use ark_poly::MultilinearExtension;
use ark_test_curves::bls12_381::Fr;

use std::fs::File;
use std::io::{Write, stdout};
use std::time::{Duration, Instant};

use crate::improved::protocol::{EvalProductSV, LinearTimeSC, SumcheckProtocol};
use crate::improved::streaming::MockStream;
use crate::utils::generate_multivariate_poly_test;

fn main() {
    let max_vars = 14; 
    let num_runs = 5;
    let degrees_to_test = [3, 6, 9]; // Added target degrees to match your python script

    println!("==================================================");
    println!("       STARTING SUMCHECK PROTOCOL BENCHMARK        ");
    println!("==================================================");

    for &d in &degrees_to_test {
        println!("\n##################################################");
        println!("  LAUNCHING BENCHMARK SERIES FOR DEGREE d = {}", d);
        println!("##################################################");
        test_range_variables(max_vars, d, num_runs);
    }

    println!("\n[GLOBAL OK] All benchmarks completed successfully!");
}

pub fn test_range_variables(max_vars: usize, d: usize, num_runs: u32) {
    let filename = format!("benchmark_results_d{}.csv", d);
    let mut file = File::create(&filename).expect("Unable to create file");

    // Header updated to match the Offline/Online approach
    writeln!(
        file,
        "Variables,Arkworks_ms,LinearTimeSC_ms,EvalProductSV_ms,EvalProductSV_Offline_ms,EvalProductSV_Online_ms"
    )
    .unwrap();
    file.flush().unwrap();

    for num_v in 4..=max_vars {
        println!("\n==================================================");
        println!(
            " Benchmarking for {} variables (2^{} = {} points, d={})",
            num_v, num_v, 1 << num_v, d
        );
        println!(" Average over {} runs...", num_runs);
        println!("==================================================");

        let mut total_ark = Duration::ZERO;
        let mut total_opt = Duration::ZERO;
        let mut total_sv_offline = Duration::ZERO;
        let mut total_sv_online = Duration::ZERO;

        for run in 1..=num_runs {
            print!("   Run {}/{}... ", run, num_runs);
            stdout().flush().unwrap();

            let (d_ark, d_opt, d_sv_offline, d_sv_online) = multivariate_test(num_v, d);

            total_ark += d_ark;
            total_opt += d_opt;
            total_sv_offline += d_sv_offline;
            total_sv_online += d_sv_online;

            println!("Done.");
        }

        let avg_ark = total_ark / num_runs;
        let avg_opt = total_opt / num_runs;
        let avg_sv_offline = total_sv_offline / num_runs;
        let avg_sv_online = total_sv_online / num_runs;
        let avg_sv_total = avg_sv_offline + avg_sv_online;

        let duration_ark_ms = avg_ark.as_secs_f64() * 1000.0;
        let duration_opt_ms = avg_opt.as_secs_f64() * 1000.0;
        let duration_sv_offline_ms = avg_sv_offline.as_secs_f64() * 1000.0;
        let duration_sv_online_ms = avg_sv_online.as_secs_f64() * 1000.0;
        let duration_sv_total_ms = avg_sv_total.as_secs_f64() * 1000.0;

        // Save all measured data
        writeln!(
            file,
            "{},{},{},{},{},{}",
            num_v, duration_ark_ms, duration_opt_ms, duration_sv_total_ms, duration_sv_offline_ms, duration_sv_online_ms
        )
        .unwrap();
        file.flush().unwrap();

        println!("\n -> Average Arkworks     : {:.4} ms", duration_ark_ms);
        println!(" -> Average LinearTimeSC  : {:.4} ms", duration_opt_ms);
        println!(" -> Average EvalProductSV : {:.4} ms (Total)", duration_sv_total_ms);
        println!("       |-- Offline (Precomp): {:.4} ms", duration_sv_offline_ms);
        println!("       |-- Online Phase     : {:.4} ms", duration_sv_online_ms);
    }
    println!("\n[OK] Series complete! Results saved in '{}'.", filename);
}

fn multivariate_test(num_vars: usize, d: usize) -> (Duration, Duration, Duration, Duration) {
    let mut rng = rand::thread_rng();
    let (list_of_poly, list_of_products) = generate_multivariate_poly_test(&mut rng, num_vars, d);

    let hypercube_size = 1 << num_vars;
    let mut expected_sum = Fr::ZERO;
    for x in 0..hypercube_size {
        let mut product_at_x = Fr::ONE;
        for k in 0..d {
            product_at_x *= list_of_poly[k].evaluations[x];
        }
        expected_sum += product_at_x;
    }

    let start_arkworks = Instant::now();
    let proof = MLSumcheck::prove(&list_of_products).expect("The Arkworks prover failed");
    let duration_arkworks = start_arkworks.elapsed();

    let ark_sum = MLSumcheck::extract_sum(&proof);
    assert_eq!(expected_sum, ark_sum, "Local hypercube sum mismatch!");

    let linear_time_protocol = LinearTimeSC;
    let l = list_of_poly[0].num_vars();
    let d_len = list_of_poly.len();
    let mut stream_opt = MockStream::new(l, d_len, &list_of_poly);

    let start_improved = Instant::now();
    let verifier_accepted_opt = linear_time_protocol.run(&mut stream_opt, expected_sum);
    let duration_improved = start_improved.elapsed();
    assert!(verifier_accepted_opt);

    let eval_product_sv_protocol = EvalProductSV::new(d_len, l);
    let mut stream_sv = MockStream::new(l, d_len, &list_of_poly);

    // Measure the Offline Phase (Pure geometric precomputation)
    let start_offline = Instant::now();
    let offline_data = eval_product_sv_protocol.precomputation_phase(&mut stream_sv);
    let duration_sv_offline = start_offline.elapsed();

    // Measure the Online Phase (Rounds simulation + Final Phase with interaction)
    let start_online = Instant::now();
    let verifier_accepted_sv = eval_product_sv_protocol.online_phase(&mut stream_sv, expected_sum, offline_data);
    let duration_sv_online = start_online.elapsed();
    assert!(verifier_accepted_sv);

    (duration_arkworks, duration_improved, duration_sv_offline, duration_sv_online)
}