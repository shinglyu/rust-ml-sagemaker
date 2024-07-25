use actix_web::{get, guard, web, App, HttpServer, HttpResponse, Responder};
use ndarray::Array;

// Decision Tree Classifier
use linfa_trees::DecisionTree;
use linfa::traits::Predict;

use std::fs::File;
use std::io::prelude::*;
use std::str;
use serde::Serialize;
use log::info;
use simple_logger::SimpleLogger;

const MODEL_PATH: &str = "/opt/ml/model/model.bincode";

#[derive(Serialize)]
struct Response {
    prediction: String,
}

struct AppData {
    model: DecisionTree<f32, String>,
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
    let x = Array::from_shape_vec((1, v.len()), v).unwrap();

    let y_hat = data.model.predict(&x);
    info!("Prediction: {:?}", y_hat);

    HttpResponse::Ok().json(Response {
        prediction: y_hat[0].to_string(),
    })
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    SimpleLogger::new().init().unwrap();
    info!("Loading model from {}", MODEL_PATH);
    let model: DecisionTree<f32, String> = {
        let mut buf: Vec<u8> = Vec::new();
        File::open(MODEL_PATH)
            .and_then(|mut f| f.read_to_end(&mut buf))
            .expect("Can not load model");
        bincode::deserialize(&buf).expect("Can not deserialize the model")
    };
    info!("Model loaded");

    let model_data = web::Data::new(AppData { model });

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