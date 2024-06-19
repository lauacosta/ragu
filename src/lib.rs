use colored::Colorize;
use pgvector::Vector;
use polars::df;
use polars::io::SerWriter;
use polars::prelude::CsvWriter;
use polars::{
    datatypes::AnyValue,
    frame::DataFrame,
    lazy::dsl::col,
    prelude::{LazyCsvReader, LazyFileListReader},
};
use reqwest::header::AUTHORIZATION;
use reqwest::Client;
use sqlx::{postgres::PgPoolOptions, Pool, Postgres};
use std::fs::File;
use std::path::{Path, PathBuf};

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
        .post(api_url.clone())
        .header(AUTHORIZATION, authorization_token.clone())
        .json(&request_body)
        .send()
        .await?;

    if res.status().is_success() {
        eprintln!("Status {:?}, Request exitoso!", res.status());
    } else if res.status().is_server_error() {
        eprintln!("Status: {:?}, Server error!", res.status());
    } else {
        eprintln!("Status: {:?}", res.status());
    }

    let res: HFResponse = match res.json().await {
        Ok(output) => HFResponse { output },
        Err(err) => anyhow::bail!(err),
    };

    Ok(res.output.into_iter().flatten().collect())
}

pub async fn generar_embeddings(
    model_id: &String,
    input: &[String],
) -> anyhow::Result<Vec<Vec<f32>>> {
    dotenvy::dotenv()?;
    let hf_token = std::env::var("HF_TOKEN").expect("No se pudo encontrar la variable 'HF_TOKEN'");
    let authorization_token = format!("Bearer {hf_token}");
    let api_url =
        format!("https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}");
    let client = Client::new();
    let mut resulting_embeddings = vec![];

    let filas_n = input.len();
    let chunk_n = 15;
    let chunks_size = filas_n / chunk_n;
    let mut i = 1;
    for chunk in input.chunks(chunks_size) {
        let request_body = HFRequest {
            inputs: chunk.to_vec(),
            options: RequestOptions {
                wait_for_model: true,
            },
        };

        let res = client
            .post(api_url.clone())
            .header(AUTHORIZATION, authorization_token.clone())
            .json(&request_body)
            .send()
            .await?;

        if res.status().is_success() {
            eprintln!(
                "({}/{}) Status {:?}, Request exitoso!",
                i.to_string().bright_green(),
                chunk_n,
                res.status()
            );
        } else if res.status().is_server_error() {
            eprintln!("Status: {:?}, Server error!", res.status());
        } else {
            eprintln!("Status: {:?}", res.status());
        }

        let res: HFResponse = match res.json().await {
            Ok(output) => HFResponse { output },
            Err(err) => anyhow::bail!(err),
        };
        resulting_embeddings.extend_from_slice(&res.output);
        i += 1;
    }

    Ok(resulting_embeddings)
}

pub fn dataframe_from_csv(path: &PathBuf) -> anyhow::Result<DataFrame> {
    Ok(LazyCsvReader::new(path)
        .with_has_header(true)
        .finish()?
        .select(&[col("*")])
        .collect()?)
}

#[allow(dead_code)]
pub async fn conectar_con_bd() -> anyhow::Result<Pool<Postgres>> {
    dotenvy::dotenv()?;
    let db_url =
        std::env::var("DATABASE_URL").expect("No se pudo encontrar la variable 'DATABASE_URL'");
    Ok(PgPoolOptions::new()
        .acquire_timeout(std::time::Duration::from_secs(4))
        .connect(&db_url)
        .await?)
}

pub async fn vectorized_dataframe(path: &Path) -> anyhow::Result<DataFrame> {
    eprintln!("Leyendo archivo CSV...");
    let df = dataframe_from_csv(&PathBuf::from(path))?;

    let mut vec_filas: Vec<String> = vec![];
    for i in 0..df.height() {
        let fila = df.get(i).unwrap();
        vec_filas.push({
            let resultado: Vec<String> = fila
                .iter()
                .map(|v| match *v {
                    AnyValue::String(s) => s.to_string(),
                    AnyValue::Int64(s) => s.to_string(),
                    AnyValue::Null => "null".to_string(),
                    _ => "Falta".to_string(),
                })
                .collect();

            resultado.join(", ")
        })
    }
    eprintln!("Archivo CSV leido!");

    eprintln!("Generando embeddings...");
    let model_id = "sentence-transformers/all-MiniLM-L6-v2".to_string();
    let vec_embeddings: Vec<String> = generar_embeddings(&model_id, &vec_filas)
        .await?
        .into_iter()
        .map(|v| {
            let parsed_f64: Vec<String> = v.iter().map(|x| x.to_string()).collect();
            let embedding_string = parsed_f64.join(", ");
            format!("[{embedding_string}]")
        })
        .collect();

    eprintln!("Embeddings generados!");
    println!("{:?}", vec_embeddings);

    eprintln!("Generando DataFrame...");

    Ok(polars::functions::concat_df_horizontal(&[
        df,
        df!("embeddings" => vec_embeddings)?,
    ])?)
}
pub async fn generar_embedded_csv(origen: &Path, destino: &Path) -> anyhow::Result<()> {
    let mut df = vectorized_dataframe(origen).await?;
    eprintln!("DataFrame generado!");

    eprintln!("Escribiendo DataFrame a archivo CSV...");
    let mut file = File::create(destino).expect("El archivo no pudo crearse.");
    CsvWriter::new(&mut file)
        .include_header(true)
        .with_separator(b',')
        .finish(&mut df)?;

    eprintln!("Archivo CSV generado!");
    Ok(())
}

#[derive(sqlx::FromRow, Debug)]
pub struct Row {
    nombre: String,
    correo: String,
    descrip: Option<String>,
    area_estudio: String,
    experiencia: Option<String>,
    estudios_recientes: Option<String>,
    puntaje: f64,
}

impl std::fmt::Display for Row {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let experiencia_str = match &self.experiencia {
            Some(exp) => exp,
            None => "No especificada",
        };

        let descrip_str = match &self.descrip {
            Some(exp) => exp,
            None => "No especificada",
        };

        let estudios_recientes_str = match &self.estudios_recientes {
            Some(exp) => exp,
            None => "No especificada",
        };

        write!(
            f,
            "Nombre: {}\nCorreo: {}\nDescripción: {}\nÁrea de Estudio: {}\nExperiencia: {}\nEstudios Recientes: {}\nPuntaje: {}",
            self.nombre,
            self.correo,
            descrip_str,
            self.area_estudio,
            experiencia_str,
            estudios_recientes_str,
            self.puntaje
        )
    }
}

pub async fn realizar_consulta(query: &String) -> anyhow::Result<()> {
    let pool = conectar_con_bd().await?;
    sqlx::migrate!("./migrations").run(&pool).await?;

    let model_id = "sentence-transformers/all-MiniLM-L6-v2";
    let embedded_query = Vector::from(embed_query(&model_id.to_string(), &query).await?);

    sqlx::query("CREATE EXTENSION IF NOT EXISTS vector")
        .execute(&pool)
        .await?;

    let result: Vec<Row> = sqlx::query_as("SELECT nombre, correo, descrip, area_estudio, experiencia, estudios_recientes, 1 - (embedding <=> $1) AS puntaje FROM usuarios ORDER BY puntaje DESC LIMIT 40").bind(embedded_query).fetch_all(&pool).await?;

    println!(
        "-----------------------------------------------------------------\n
        QUERY: {}
        \n-----------------------------------------------------------------",
        query
    );
    for res in result {
        println!(
            "{}\n-----------------------------------------------------------------\n",
            res
        );
    }
    Ok(())
}
