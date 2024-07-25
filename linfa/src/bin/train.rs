use std::error::Error;
use std::fs::File;
use std::io::prelude::*;

use linfa::prelude::*;
use linfa_trees::{DecisionTree, Result, SplitQuality};
use ndarray_rand::rand::{SeedableRng};
use rand::rngs::SmallRng;
use log::info;
use simple_logger::SimpleLogger;

const TRAIN_DATA_PATH: &str = "opt/ml/input/data/training/iris.csv";
const MODEL_OUTPUT_PATH: &str = "/opt/ml/model/";

// How to use this example script:
// 1. Replace main() function with your own training logic
// 2. Read the training dataset from TRAIN_DATA_PATH
// 3. Write your model artifact to MODEL_OUTPUT_PATH

// This example is taken from https://github.com/rust-ml/linfa/blob/master/algorithms/linfa-trees/examples/decision_tree.rs
fn main() -> Result<()> {
    SimpleLogger::new().init().unwrap();

    let mut rng = SmallRng::seed_from_u64(42);
    //TODO: read dataset from file
    let (train, test) = linfa_datasets::iris().shuffle(&mut rng).split_with_ratio(0.8);

    println!("Training model with Gini criterion ...");
    let gini_model = DecisionTree::params()
        .split_quality(SplitQuality::Gini)
        .max_depth(Some(100))
        .min_weight_split(1.0)
        .min_weight_leaf(1.0)
        .fit(&train)?;

    let gini_pred_y = gini_model.predict(&test);
    let cm = gini_pred_y.confusion_matrix(&test)?;

    println!("{:?}", cm);
    println!(
        "Test accuracy with Gini criterion: {:.2}%",
        100.0 * cm.accuracy()
    );

    let feats = gini_model.features();
    println!("Features trained in this tree {:?}", feats);

    /*
    println!("Training model with entropy criterion ...");
    let entropy_model = DecisionTree::params()
        .split_quality(SplitQuality::Entropy)
        .max_depth(Some(100))
        .min_weight_split(10.0)
        .min_weight_leaf(10.0)
        .fit(&train)?;

    let entropy_pred_y = entropy_model.predict(&test);
    let cm = entropy_pred_y.confusion_matrix(&test)?;

    println!("{:?}", cm);
    println!(
        "Test accuracy with Entropy criterion: {:.2}%",
        100.0 * cm.accuracy()
    );

    let feats = entropy_model.features();
    println!("Features trained in this tree {:?}", feats);
    */

    // Export model
    {
        info!("Saving model artifact to {}", MODEL_OUTPUT_PATH);
        let model_bytes = bincode::serialize(&gini_model).expect("Can not serialize the model");
        File::create(format!("{}/model.bincode", MODEL_OUTPUT_PATH)) // TODO use Path
            .and_then(|mut f| f.write_all(&model_bytes))
            .expect("Can not persist model");
        info!("Model saved");
    }
    // Exporting the tree diagram
    let mut tikz = File::create(format!("{}/decision_tree_example.tex", MODEL_OUTPUT_PATH)).unwrap();
    tikz.write_all(
        gini_model
            .export_to_tikz()
            .with_legend()
            .to_string()
            .as_bytes(),
    )
    .unwrap();
    println!(" => generate Gini tree description with `latex decision_tree_example.tex`!");



    Ok(())
}