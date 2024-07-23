use actix_web::{get, post, guard, web, App, HttpServer, HttpResponse, Responder};
use ndarray::Array;
use smartcore::dataset::*;
// Random Forest
use smartcore::tree::decision_tree_classifier::DecisionTreeClassifier;
// Model performance
use smartcore::metrics::mean_squared_error;
use smartcore::model_selection::train_test_split;
use std::error::Error;
use std::fs::File;
use std::io::prelude::*;
use std::str;
use serde::Serialize;
use log::info;
use simple_logger::SimpleLogger;


const MODEL_PATH: &str ="/opt/ml/model/model.bincode";
//const MODEL_PATH: &str ="/tmp/model.bincode";


#[derive(Serialize)]
struct Response {
    prediction: f32
}

struct AppData {
    model: DecisionTreeClassifier<f32>
}

#[get("/ping")]
async fn ping() -> impl Responder {
    HttpResponse::Ok()
}

async fn invocations(body: web::Bytes, data: web::Data<AppData>) -> impl Responder {
    let csv = str::from_utf8(&body).unwrap();
    let v: Vec<f32> = csv
        .split(',')
        .map(|s| s.parse().expect("not floating number"))
        .collect();
    info!("Received: {:?}", v);
    let x = Array::from_shape_vec( (1, v.len() ), v).unwrap();
    let y_hat = data.model.predict(&x).unwrap();
    info!("Prediction: {:?}", y_hat);
    HttpResponse::Ok().json(Response{
        prediction: y_hat[0] as f32,
    })
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    SimpleLogger::new().init().unwrap();
    info!("Loading model from {}", MODEL_PATH);
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
                    .guard(guard::Header("content-type", "text/csv"))
                    .to(invocations),
            )
    })
    .bind(("0.0.0.0", 8080))?
    .run()
    .await
}