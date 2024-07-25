use std::error::Error;
use std::fs::File;
use std::io::prelude::*;

use ndarray::Array;
use ndarray::prelude::*;
use ndarray_csv::Array2Reader;

use linfa::prelude::*;
use linfa_trees::{DecisionTree, SplitQuality};
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
fn main() -> Result<(), Box<dyn Error>> {
    SimpleLogger::new().init().unwrap();

    info!("Loading dataset");
    let mut reader = csv::ReaderBuilder::new()
       .has_headers(false)
       .from_path(TRAIN_DATA_PATH)
       .expect(&format!("Fail to read the CSV at {}", TRAIN_DATA_PATH));

       
    let data: Array2<String> = reader.deserialize_array2_dynamic()?;
    
    // Parsing the training dataset into features (x) and target (y)
    let x_strings = data.slice(s![.., 0..4]);
    let parsed_x_strings: Vec<_> = x_strings
        .into_iter()
        .flat_map(|s| s.parse::<f32>().ok())
        .collect();
    let x = Array::from_shape_vec((data.nrows(), 4), parsed_x_strings)?;
    
    let y_array_view = data.slice(s![.., 4]);
    let y: Array<&str, _> = y_array_view.iter().map(|s| s.as_str()).collect();
    
    let data = Dataset::new(x, y); // TODO: with feature names // TODO use map_targets
    
    let mut rng = SmallRng::seed_from_u64(42);
    let (train, test) = data.shuffle(&mut rng).split_with_ratio(0.8);
    info!("Dataset loaded and preprocessed");

    info!("Training model with Gini criterion ...");
    let gini_model = DecisionTree::params()
        .split_quality(SplitQuality::Gini)
        .max_depth(Some(100))
        .min_weight_split(1.0)
        .min_weight_leaf(1.0)
        .fit(&train)?;

    // Optional: showing that the model works
    let gini_pred_y = gini_model.predict(&test);
    let cm = gini_pred_y.confusion_matrix(&test)?;

    info!("{:?}", cm);
    info!(
        "Test accuracy with Gini criterion: {:.2}%",
        100.0 * cm.accuracy()
    );

    let feats = gini_model.features();
    info!("Features trained in this tree {:?}", feats);


    // Export model
    {
        info!("Saving model artifact to {}", MODEL_OUTPUT_PATH);
        let model_bytes = bincode::serialize(&gini_model).expect("Can not serialize the model");
        File::create(format!("{}/model.bincode", MODEL_OUTPUT_PATH)) // TODO use Path
            .and_then(|mut f| f.write_all(&model_bytes))
            .expect("Can not persist model");
        info!("Model saved");
    }

    // Optional:  Exporting the tree diagram
    let mut tikz = File::create(format!("{}/decision_tree_example.tex", MODEL_OUTPUT_PATH)).unwrap();
    tikz.write_all(
        gini_model
            .export_to_tikz()
            .with_legend()
            .to_string()
            .as_bytes(),
    )
    .unwrap();
    info!(" => generate Gini tree description with `latex decision_tree_example.tex`!");

    Ok(())
}