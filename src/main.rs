use anyhow::bail;
use clap::Subcommand;
use clap::{Args, Parser};
use pgvector::Vector;
use std::path::PathBuf;
use tracing::{error, info, span, Instrument, Level};

#[derive(Parser)]
#[clap(
    name = "ragu",
    version = "0.1",
    author = "Lautaro Acosta Quintana",
    about = "Una CLI para busquedas semánticas en una base de datos."
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    #[clap(name = "ask", about = "Realiza una consulta a la base de datos.")]
    Ask(AskArgs),

    #[clap(name = "load", about = "Cargar un documento .csv a la base de datos.")]
    Load(LoadArgs),

    #[clap(
        name = "export_to_csv",
        about = "Genera embeddings a partir de un archivo .csv y los exporta a otro .csv."
    )]
    Csv(AskArgs),
}

#[derive(Args)]
struct AskArgs {
    #[clap(
        short,
        long,
        help = "Consulta sobre la cual realizar la busqueda semántica."
    )]
    consulta: String,

    #[clap(
        long = "ctx",
        help = "Contexto con el cual extraer información de la base de datos y realizar la consulta."
    )]
    contexto: Option<String>,
}

#[derive(Args)]
struct LoadArgs {
    #[clap(help = "archivo fuente.")]
    #[arg(short, long)]
    file: PathBuf,
}

#[derive(Args)]
struct CsvArgs {
    #[clap(help = "archivo fuente.")]
    #[arg(short, long)]
    file: PathBuf,
    #[clap(help = "donde se almancenará el resultado.")]
    #[arg(short, long)]
    destino: PathBuf,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let pool = ragu::conectar_con_bd().await?;
    sqlx::migrate!("./migrations").run(&pool).await?;

    tracing_subscriber::fmt::init();
    let span = span!(Level::INFO, "raiduki");
    let _guard = span.enter();

    let cli = Cli::parse();

    match &cli.command {
        Commands::Ask(args) => {
            if let Err(err) =
                ragu::semantic_search(&args.consulta, args.contexto.as_deref(), &pool).await
            {
                error!("{}", err.to_string());
                bail!(err);
            }
        }
        Commands::Load(args) => {
            let (data, embeddings) = ragu::vectorize_csv(&args.file).await?;
            if let Err(err)  = sqlx::query!("INSERT INTO datos_usuarios (data, embedding) SELECT * FROM UNNEST($1::text[], $2::vector[])", &data[..], &embeddings[..] as &[Vector]).execute(&pool).in_current_span().await {
                error!("{}",err.to_string());
                bail!(err);
            };
            info!("Se han cargado los datos exitosamente!");
        }
        Commands::Csv(args) => {
            // let candle_embed = CandleEmbedBuilder::new().build()?;

            // let embeddings = candle_embed.embed_one(&args.consulta)?;

            // dbg!("{}", embeddings);
        }
    };
    Ok(())
}
