use ctm_demo::{Dahlquist, Model, ModelType};
use plotpy::{Curve, Plot, linspace};
use russell_lab::approx_eq;
use russell_ode::Method;
use std::collections::HashMap;

const SAVE_FIGURE: bool = false;

#[test]
fn test_dahlquist() {
    // Allocate the model
    let lambda = 5.0;
    let method = Method::DoPri5;
    let mut model = Model::new(ModelType::Dahlquist, HashMap::from([("lambda", lambda)]), method).unwrap();

    // Set initial conditions
    let x_ini = 0.0;
    let y_ini = Dahlquist::analytical_y(lambda, x_ini);

    // Define constants for the backward Euler update
    let ddx = 0.1;
    let nd = 5;

    // Perform the backward Euler update
    let (xx, yy, yy_ode, _, ctm_list, num_ctm_list, num_ctm_ode_list) = model.simulate(x_ini, y_ini, ddx, nd).unwrap();

    // Generate the plot
    if SAVE_FIGURE {
        // Fine y(x) curve
        let xx_fine = linspace(0.0, 0.5, 101);
        let yy_fine = xx_fine
            .iter()
            .map(|&x| Dahlquist::analytical_y(lambda, x))
            .collect::<Vec<_>>();
        let mut curve_fine = Curve::new();
        curve_fine.set_label("Analytical Solution").draw(&xx_fine, &yy_fine);

        // ODE solution
        let mut curve_ode = Curve::new();
        curve_ode
            .set_label(&format!("{:?} solution", method))
            .set_line_style("--")
            .draw(&xx, &yy_ode);

        // Continuous modulus curve
        let com_list = xx_fine
            .iter()
            .enumerate()
            .map(|(i, &x)| model.continuous_modulus(x, yy_fine[i]))
            .collect::<Vec<_>>();
        let mut curve_com = Curve::new();
        curve_com.set_label("Continuous Modulus").draw(&xx_fine, &com_list);

        // Backward Euler curve y(x)
        let mut curve = Curve::new();
        curve
            .set_label("Backward Euler")
            .set_line_style("None")
            .set_marker_style("o")
            .draw(&xx, &yy);

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
            .add(&curve_fine)
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
            .save("/tmp/consistent_tangent/test_dahlquist.svg")
            .unwrap();
    }

    // Check the results against Mathematica results
    let yy_ref = [
        1.0,
        0.6666666666666666,
        0.4444444444444444,
        0.2962962962962963,
        0.19753086419753085,
        0.1316872427983539,
    ];
    for i in 0..nd + 1 {
        approx_eq(yy[i], yy_ref[i], 1e-15);
    }

    // Check the results against Mathematica results
    let ctm_ref = [
        -5.0,
        -2.2222222222222223,
        -1.4814814814814812,
        -0.9876543209876543,
        -0.6584362139917694,
        -0.43895747599451296,
    ];
    for i in 0..nd + 1 {
        approx_eq(ctm_list[i], ctm_ref[i], 1e-15);
    }

    // Compare the consistent tangent moduli
    for i in 0..nd + 1 {
        approx_eq(ctm_list[i], num_ctm_list[i], 1e-4);
    }
}
