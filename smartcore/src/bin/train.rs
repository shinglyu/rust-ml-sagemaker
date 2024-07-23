use ndarray::Array;
use ndarray::prelude::*;
use ndarray_csv::Array2Reader;
// Random Forest
use smartcore::tree::decision_tree_classifier::DecisionTreeClassifier;
// Model performance
use std::error::Error;
use std::fs::File;
use std::io::prelude::*;
use log::info;
use simple_logger::SimpleLogger;
use csv;


const TRAIN_DATA_PATH: &str = "opt/ml/input/data/training/iris.csv";
//const TRAIN_DATA_PATH: &str = "./dataset/iris.csv";
const MODEL_OUTPUT_PATH: &str ="/opt/ml/model/model.bincode";
//const MODEL_OUTPUT_PATH: &str ="/tmp/model.bincode"; // For local testing
// TODO: failure ouptut path

fn main() -> Result<(), Box<dyn Error>> {
    SimpleLogger::new().init().unwrap();
    info!("Loading dataset");
    // Load dataset
    //let iris_data = iris::load_dataset();
    // Read the CSV
    let mut reader = csv::ReaderBuilder::new()
       .has_headers(false)
       .from_path(TRAIN_DATA_PATH)
       .expect(&format!("Fail to read the CSV at {}", TRAIN_DATA_PATH));
    let data: Array2<String> = reader.deserialize_array2_dynamic()?;
    
    let x_strings = data.slice(s![..,0..4]);
    let mut vec_x: Vec<f32> = Vec::new();
    for i in x_strings.iter() {
        println!("{}", i);
        vec_x.push(i.parse().unwrap());
    }
    let x = Array::from_shape_vec( (data.nrows(), 4), vec_x )?;
    let y_strings = data.slice(s![.., 4]);
    let mut vec_y: Vec<f32> = Vec::new();
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
    // Transform dataset into a NxM matrix
    /*
    let x = Array::from_shape_vec(
        (iris_data.num_samples, iris_data.num_features),
        iris_data.data,
    )?;
    */
    // These are our target class labels
    /*
    let y = Array::from_shape_vec(iris_data.num_samples, iris_data.target)?;
    */
    // Split dataset into training/test (80%/20%)
    //let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, true);
    //println!("{}", x_test);
    info!("Dataset loaded and preprocessed");
    info!("Training model");
    // Random Forest
    let rf_model = DecisionTreeClassifier::fit(&x, &y, Default::default())?;
        // .and_then(|rf| rf.predict(&x_test)).unwrap();
    // Calculate test error
    /*
    println!("MSE: {}", mean_squared_error(&y_test, &y_hat_rf));
    println!("{:?}", &y_test);
    println!("{:?}", &y_hat_rf);
    println!("Diff: {}", &y_hat_rf- &y_test);
    */
    
    // Save the model
    info!("Model training succeede");
    {
        info!("Saving model to {}", MODEL_OUTPUT_PATH);
        let model_bytes = bincode::serialize(&rf_model).expect("Can not serialize the model");
        File::create(MODEL_OUTPUT_PATH)
            .and_then(|mut f| f.write_all(&model_bytes))
            .expect("Can not persist model");
        info!("Model saved");
    }
    
    Ok(())
}
