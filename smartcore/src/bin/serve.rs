use actix_web::{get, guard, web, App, HttpServer, HttpResponse, Responder};
use ndarray::Array;

// Random Forest
use smartcore::tree::decision_tree_classifier::DecisionTreeClassifier;

use std::fs::File;
use std::io::prelude::*;
use std::str;
use serde::Serialize;
use log::info;
use simple_logger::SimpleLogger;


const MODEL_PATH: &str ="/opt/ml/model/model.bincode";

#[derive(Serialize)]
struct Response {
    prediction: f32
}

struct AppData {
    // Change the model here as needed
    model: DecisionTreeClassifier<f32>
}

#[get("/ping")]
async fn ping() -> impl Responder {
    HttpResponse::Ok()
}

// Change this invocations function to 
// 1. Parse the inference payload
// 2. Send it to the data.model for prediction
async fn invocations(body: web::Bytes, data: web::Data<AppData>) -> impl Responder {
    // Parsing the inference payload\
    let csv = str::from_utf8(&body).unwrap();
    let v: Vec<f32> = csv
        .split(',')
        .map(|s| s.parse().expect("not floating number"))
        .collect();
    info!("Received: {:?}", v);
    let x = Array::from_shape_vec( (1, v.len() ), v).unwrap();
    
    // Prediction
    let y_hat = data.model.predict(&x).unwrap();
    info!("Prediction: {:?}", y_hat);

    // Send the response
    HttpResponse::Ok().json(Response{
        prediction: y_hat[0] as f32,
    })
}

// Change this main function to load the model
// The model is loaded here instaed of in the invocation function so it can be reused across invocations 
#[actix_web::main]
async fn main() -> std::io::Result<()> {
    SimpleLogger::new().init().unwrap();
    info!("Loading model from {}", MODEL_PATH);
    // Change the model as needed
    let model: DecisionTreeClassifier<f32> = {
        let mut buf: Vec<u8> = Vec::new();
        File::open(MODEL_PATH)
            .and_then(|mut f| f.read_to_end(&mut buf))
            .expect("Can not load model");
        bincode::deserialize(&buf).expect("Can not deserialize the model")
    };
    info!("Model loaded");
    
    let model_data = web::Data::new(AppData {
        model: model
    });
    
    info!("Starting server on port 8080");
    HttpServer::new(move || {
        App::new()
            .app_data(model_data.clone())
            .service(ping)
            .route(
                "/invocations",
                web::post()
                    .guard(guard::Header("content-type", "text/csv")) // Change the content type as needed
                    .to(invocations),
            )
    })
    .bind(("0.0.0.0", 8080))?
    .run()
    .await
}