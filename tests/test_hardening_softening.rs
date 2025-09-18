use ctm_demo::{Model, ModelType};
use plotpy::{Curve, Plot};
use russell_lab::approx_eq;
use russell_ode::Method;
use std::collections::HashMap;

const SAVE_FIGURE: bool = true;

// Mathematica reference values
#[rustfmt::skip]
const MATHEMATICA_XX: [f64; 51] = [
    0., 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17,
    0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3, 0.31, 0.32, 0.33, 0.34, 0.35,
    0.36, 0.37, 0.38, 0.39, 0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5,
];

// Mathematica reference values
#[rustfmt::skip]
const MATHEMATICA_YY: [f64;51] = [
    0., 0.0921923529874959, 0.180995936702942, 0.265135546339778, 0.34303769588157, 0.412909639654633, 0.472933902640526, 0.521575383562834, 0.557924601234833, 0.581937761631112, 0.594452730469534, 0.596972402157104, 0.591326450808571, 0.579354305707458,
    0.562695563124671, 0.542698423576125, 0.520413194773086, 0.496630461299565, 0.471934147069921, 0.446753034002368, 0.421403915953387, 0.39612495266096, 0.371100159763253, 0.346476741387926, 0.322376942591868, 0.298905838568251, 0.276156158431712, 0.254210961837679,
    0.23314478624825, 0.213023733754134, 0.193904874033716, 0.175835268002485, 0.158850897482815, 0.142975694678168, 0.128220878500363, 0.114584707614984, 0.102052740913371, 0.0905985737010805, 0.0801850545040245, 0.070765847217885, 0.062287241401217, 0.0546900747507887,
    0.047911642057484, 0.0418874859643884, 0.0365529932436868, 0.0318447432166188, 0.0277015895758155, 0.0240654765039555, 0.0208820056359409, 0.0181007846054456, 0.0156755917049248,
];

fn run_test(name: &str, first: usize, x_ini: f64, y_ini: f64, ddx: f64, nd: usize) {
    // Allocate the model
    let method = Method::DoPri5;
    let mut model = Model::new(
        ModelType::HardeningSoftening,
        HashMap::from([("li", 10.0), ("lr", 3.0), ("y0r", 1.0), ("a", 3.0), ("b", 5.0)]),
        method,
    )
    .unwrap();

    // Perform the backward Euler update
    let (xx, yy, yy_ode, com_list, ctm_list, num_ctm_list, num_ctm_ode_list) =
        model.simulate(x_ini, y_ini, ddx, nd).unwrap();

    // Generate the plot
    if SAVE_FIGURE {
        let mut curve_ref = Curve::new();
        curve_ref
            .set_label("Mathematica")
            .draw(&MATHEMATICA_XX, &MATHEMATICA_YY);

        // ODE solution
        let mut curve_ode = Curve::new();
        curve_ode
            .set_label(&format!("{:?} solution", method))
            .set_line_style("--")
            .draw(&xx, &yy_ode);

        // Backward Euler curve y(x)
        let mut curve = Curve::new();
        curve
            .set_label("Backward Euler")
            .set_line_style("None")
            .set_marker_style(".")
            .draw(&xx, &yy);

        // Continuous modulus curve
        let mut curve_com = Curve::new();
        curve_com.set_label("Continuous Modulus").draw(&xx, &com_list);

        // Consistent tangent modulus curve
        let mut curve_ctm = Curve::new();
        curve_ctm
            .set_label("Consistent Tangent Modulus")
            .set_line_style("None")
            .set_marker_style(".")
            .draw(&xx, &ctm_list);

        // Numerical consistent tangent modulus curve (BE version)
        let mut curve_ctm_num = Curve::new();
        curve_ctm_num
            .set_label("Numerical CTM (BE)")
            .set_line_style("None")
            .set_marker_void(true)
            .set_marker_size(5.0)
            .set_marker_style("o")
            .set_marker_line_color("black")
            .draw(&xx, &num_ctm_list);

        // Numerical consistent tangent modulus curve (ODE version)
        let mut curve_ctm_ode_num = Curve::new();
        curve_ctm_ode_num
            .set_label(&format!("Numerical CTM ({:?})", method))
            .set_line_style("None")
            .set_marker_style("*")
            .draw(&xx, &num_ctm_ode_list);

        // Generate the plot
        let mut plot = Plot::new();
        plot.set_subplot(1, 2, 1)
            .add(&curve_ref)
            .add(&curve_ode)
            .add(&curve)
            .grid_labels_legend("x", "y")
            .set_subplot(1, 2, 2)
            .add(&curve_com)
            .add(&curve_ctm)
            .add(&curve_ctm_num)
            .add(&curve_ctm_ode_num)
            .grid_labels_legend("x", "D")
            .set_figure_size_points(800.0, 300.0)
            .save(&format!("/tmp/ctm_demo/{}.svg", name))
            .unwrap();
    }

    // Check the results against Mathematica results
    if ddx > 0.0 {
        for i in 0..nd + 1 {
            approx_eq(yy[i], MATHEMATICA_YY[i], 0.022);
        }
    } else {
        for j in 0..nd + 1 {
            approx_eq(xx[j], MATHEMATICA_XX[first - j], 1e-15);
            approx_eq(yy[j], MATHEMATICA_YY[first - j], 0.2);
            // println!("x = {}, y = {} ({})", xx[j], yy[j], MATHEMATICA_YY[first - j]);
        }
    }

    // Compare the consistent tangent moduli
    for i in 0..nd + 1 {
        // println!("i = {}, x = {}, ctm = {}, num_ctm = {}", i, xx[i], ctm_list[i], num_ctm_list[i]);
        let tol = if ddx > 0.0 {
            if i < 26 { 0.001 } else { 0.03 }
        } else {
            0.002
        };
        approx_eq(ctm_list[i], num_ctm_list[i], tol);
    }
}

#[test]
fn test_hardening_softening_curve() {
    let x_ini = 0.0;
    let y_ini = 0.0;
    let ddx = 0.01;
    let nd = 50;
    run_test("test_hardening_softening_forward", 0, x_ini, y_ini, ddx, nd);

    let first = 10;
    let x_ini = MATHEMATICA_XX[first];
    let y_ini = MATHEMATICA_YY[first];
    let ddx = -0.01;
    let nd = 9;
    run_test("test_hardening_softening_backward", first, x_ini, y_ini, ddx, nd);
}

#[test]
fn test_hardening_softening_curve_coarse() {
    // Allocate the model
    let method = Method::DoPri5;
    let mut model = Model::new(
        ModelType::HardeningSoftening,
        HashMap::from([("li", 10.0), ("lr", 3.0), ("y0r", 1.0), ("a", 3.0), ("b", 5.0)]),
        method,
    )
    .unwrap();

    // Set initial conditions
    let x_ini = 0.0;
    let y_ini = 0.0;

    // Define constants for the backward Euler update
    let ddx = 0.05;
    let nd = 10;

    // Perform the backward Euler update
    let (xx, yy, yy_ode, com_list, ctm_list, num_ctm_list, num_ctm_ode_list) =
        model.simulate(x_ini, y_ini, ddx, nd).unwrap();

    // Generate the plot
    if SAVE_FIGURE {
        // ODE solution
        let mut curve_ode = Curve::new();
        curve_ode
            .set_label(&format!("{:?} solution", method))
            .set_line_style("--")
            .draw(&xx, &yy_ode);

        // Backward Euler curve y(x)
        let mut curve = Curve::new();
        curve.set_label("Backward Euler").set_marker_style("o").draw(&xx, &yy);

        // Continuous modulus curve
        let mut curve_com = Curve::new();
        curve_com.set_label("Continuous Modulus").draw(&xx, &com_list);

        // Consistent tangent modulus curve
        let mut curve_ctm = Curve::new();
        curve_ctm
            .set_label("Consistent Tangent Modulus")
            .set_line_style("None")
            .set_marker_style("o")
            .draw(&xx, &ctm_list);

        // Numerical consistent tangent modulus curve (BE version)
        let mut curve_ctm_num = Curve::new();
        curve_ctm_num
            .set_label("Numerical CTM (BE)")
            .set_line_style("None")
            .set_marker_void(true)
            .set_marker_size(5.0)
            .set_marker_style("o")
            .set_marker_line_color("black")
            .draw(&xx, &num_ctm_list);

        // Numerical consistent tangent modulus curve (ODE version)
        let mut curve_ctm_ode_num = Curve::new();
        curve_ctm_ode_num
            .set_label(&format!("Numerical CTM ({:?})", method))
            .set_line_style("None")
            .set_marker_style("*")
            .draw(&xx, &num_ctm_ode_list);

        // Generate the plot
        let mut plot = Plot::new();
        plot.set_subplot(1, 2, 1)
            .add(&curve_ode)
            .add(&curve)
            .grid_labels_legend("x", "y")
            .set_subplot(1, 2, 2)
            .add(&curve_com)
            .add(&curve_ctm)
            .add(&curve_ctm_num)
            .add(&curve_ctm_ode_num)
            .grid_labels_legend("x", "D")
            .set_figure_size_points(800.0, 300.0)
            .save("/tmp/ctm_demo/test_hardening_softening_coarse.svg")
            .unwrap();
    }

    // Compare the consistent tangent moduli
    for i in 0..nd + 1 {
        // println!("i = {}, x = {}, ctm = {}, num_ctm = {}", i, xx[i], ctm_list[i], num_ctm_list[i]);
        let tol = if i < 6 { 0.001 } else { 0.4 };
        approx_eq(ctm_list[i], num_ctm_list[i], tol);
    }
}
