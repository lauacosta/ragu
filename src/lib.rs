use colored::Colorize;
use pgvector::Vector;
use reqwest::header::AUTHORIZATION;
use reqwest::Client;
use sqlx::{postgres::PgPoolOptions, Pool, Postgres};
use std::path::Path;
use tracing::{debug, error, info, warn};

pub async fn conectar_con_bd() -> anyhow::Result<Pool<Postgres>> {
    dotenvy::dotenv()?;
    let db_url =
        std::env::var("DATABASE_URL").expect("No se pudo encontrar la variable 'DATABASE_URL'");
    Ok(PgPoolOptions::new()
        .acquire_timeout(std::time::Duration::from_secs(4))
        .connect(&db_url)
        .await?)
}

#[derive(serde::Serialize, serde::Deserialize)]
struct RequestOptions {
    wait_for_model: bool,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct HFRequest {
    inputs: Vec<String>,
    options: RequestOptions,
}

#[derive(serde::Serialize, serde::Deserialize, Debug)]
struct HFResponse {
    output: Vec<Vec<f32>>,
}

pub async fn embed_query(model_id: &String, input: &String) -> anyhow::Result<Vec<f32>> {
    dotenvy::dotenv()?;
    let hf_token = std::env::var("HF_TOKEN").expect("No se pudo encontrar la variable 'HF_TOKEN'");
    let authorization_token = format!("Bearer {hf_token}");
    let api_url =
        format!("https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}");
    let client = Client::new();
    let request_body = HFRequest {
        inputs: vec![input.to_string()],
        options: RequestOptions {
            wait_for_model: true,
        },
    };

    let res = client
        .post(api_url)
        .header(AUTHORIZATION, authorization_token)
        .json(&request_body)
        .send()
        .await?;

    if res.status().is_success() {
        debug!("Status {:?}, Request exitoso!", res.status());
    } else if res.status().is_server_error() {
        error!("Status: {:?}, Server error!", res.status());
    } else {
        warn!("Status: {:?}", res.status());
    }

    let res: HFResponse = match res.json().await {
        Ok(output) => HFResponse { output },
        Err(err) => anyhow::bail!(err),
    };

    Ok(res.output.into_iter().flatten().collect())
}

async fn request_embeddings(
    token: &str,
    api_url: &str,
    client: &Client,
    body: HFRequest,
) -> anyhow::Result<HFResponse> {
    let res = client
        .post(api_url)
        .header(AUTHORIZATION, format!("Bearer {token}"))
        .json(&body)
        .send()
        .await?;

    assert!(res.status().is_success());

    let res: HFResponse = match res.json().await {
        Ok(output) => HFResponse { output },
        Err(err) => anyhow::bail!(err),
    };

    Ok(res)
}

pub async fn gen_embeddings(model_id: &str, input: &[String]) -> anyhow::Result<Vec<Vector>> {
    dotenvy::dotenv()?;
    let hf_token = std::env::var("HF_TOKEN").expect("No se pudo encontrar la variable 'HF_TOKEN'");
    let api_url =
        format!("https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}");
    let client = Client::new();
    let mut resulting_embeddings: Vec<Vector> = vec![];

    let filas_n = input.len();
    if filas_n > 2000 {
        let chunk_size = 500;
        let chunk_n = filas_n / chunk_size;
        let mut i = 1;
        info!("Dividiendo la entrada en {chunk_n} chunks de {chunk_size} filas.");

        for chunk in input.chunks(chunk_size) {
            let request_body = HFRequest {
                inputs: chunk.to_vec(),
                options: RequestOptions {
                    wait_for_model: true,
                },
            };
            let response = request_embeddings(&hf_token, &api_url, &client, request_body).await?;
            let aux: Vec<Vector> = response.output.into_iter().map(Vector::from).collect();
            resulting_embeddings.extend_from_slice(&aux);
            info!(cargados = format!("({i}/{chunk_n})"), "cargando chunks.");
            i += 1;
        }
    } else {
        let request_body = HFRequest {
            inputs: input.to_vec(),
            options: RequestOptions {
                wait_for_model: true,
            },
        };

        let response = request_embeddings(&hf_token, &api_url, &client, request_body).await?;

        let aux: Vec<Vector> = response.output.into_iter().map(Vector::from).collect();
        resulting_embeddings.extend_from_slice(&aux);
    }

    Ok(resulting_embeddings)
}

pub fn read_from_csv<P: AsRef<Path>>(path: P) -> anyhow::Result<Vec<String>> {
    let path = std::fs::File::open(path)?;
    let buffer = std::io::BufReader::new(path);
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .delimiter(b',')
        .flexible(true)
        .trim(csv::Trim::All)
        .from_reader(buffer);

    let mut resultado = vec![];
    for r in reader.records() {
        let valor = r?;
        let req: Vec<String> = valor.into_iter().map(|str| str.to_string()).collect();
        let string = req.join(", ");
        resultado.push(string);
    }
    Ok(resultado)
}

pub async fn vectorize_csv<P: AsRef<Path>>(path: P) -> anyhow::Result<(Vec<String>, Vec<Vector>)> {
    info!("Leyendo archivo CSV...");
    let data = read_from_csv(path)?;
    info!("Archivo CSV leido!");

    info!("Generando embeddings...");
    let model_id = "sentence-transformers/all-MiniLM-L6-v2".to_string();
    let vec_embeddings = gen_embeddings(&model_id, &data).await?;
    info!("Embeddings generados!");

    Ok((data, vec_embeddings))
}

#[derive(sqlx::FromRow, Debug, serde::Serialize)]
pub struct Row {
    data: String,
    puntaje: Option<f64>,
}

pub async fn semantic_search(query: &String, pool: &Pool<Postgres>) -> anyhow::Result<()> {
    let model_id = "sentence-transformers/all-MiniLM-L6-v2";
    let embedded_query = Vector::from(embed_query(&model_id.to_string(), query).await?);

    // sqlx::query("CREATE EXTENSION IF NOT EXISTS vector")
    //     .execute(pool)
    //     .await?;

    let result = sqlx::query_as!(
        Row,
        "SELECT data, 1 - (embedding <=> $1) AS puntaje FROM datos_usuarios ORDER BY puntaje DESC",
        embedded_query as Vector
    )
    .fetch_all(pool)
    .await?;

    let json = serde_json::to_string_pretty(&result)?;
    println!("{}", json);

    Ok(())
}
