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
    
    /*
    let y_array_view = data.slice(s![.., 4]);
    let y: Array<String, _> = y_array_view.iter().map(|s| s.to_string()).collect();
    */
    let y_strings = data.slice(s![.., 4]);
    let mut vec_y: Vec<&str> = Vec::new();
    for t in y_strings.iter() {
        vec_y.push(t);
    }
    let y = Array::from_shape_vec(data.nrows(), vec_y).unwrap();


    
    info!("Dataset loaded and preprocessed");
    let data = Dataset::new(x, y); // TODO: with feature names // TODO use map_targets
    // Parsing the training dataset into features (x) and target (y)
    /*
    let x_strings = data.slice(s![..,0..4]);
    let mut vec_x: Vec<f64> = Vec::new();
    for i in x_strings.iter() {
        vec_x.push(i.parse().unwrap());
    }
    let x = Array::from_shape_vec( (data.nrows(), 4), vec_x )?;
    let y_strings = data.slice(s![.., 4]);
    let mut vec_y: Vec<f64> = Vec::new();
    for t in y_strings.iter() {
        let t_f = match t.as_str() {
            "Iris-setosa" => 0.,
            "Iris-versicolor" => 1.,
            "Iris-virginica" => 2.,
            _ => 0.,
        };
        vec_y.push(t_f);
    }
    let y = Array::from_shape_vec(data.nrows(), vec_y).unwrap();
    */
    
    info!("Dataset loaded and preprocessed");
    let mut rng = SmallRng::seed_from_u64(42);
    let (train, test) = data.shuffle(&mut rng).split_with_ratio(0.8);

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