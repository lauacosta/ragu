use anyhow::bail;
use clap::Subcommand;
use clap::{Args, Parser};
use pgvector::Vector;
use std::path::PathBuf;
use tracing::{error, info, span, Level};

#[derive(Parser)]
#[clap(
    name = "raiduki",
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
    Csv(CsvArgs),
}

#[derive(Args)]
struct AskArgs {
    #[clap(
        short,
        long,
        help = "Consulta sobre la cual realizar la busqueda semántica."
    )]
    consulta: String,
}

#[derive(Args)]
struct LoadArgs {
    #[clap(help = "archivo fuente.")]
    #[arg(short, long)]
    origen: PathBuf,
}

#[derive(Args)]
struct CsvArgs {
    #[clap(help = "archivo fuente.")]
    #[arg(short, long)]
    origen: PathBuf,
    #[clap(help = "donde se almancenará el resultado.")]
    #[arg(short, long)]
    destino: PathBuf,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let pool = raiduki::conectar_con_bd().await?;
    sqlx::migrate!("./migrations").run(&pool).await?;

    tracing_subscriber::fmt::init();
    let span = span!(Level::INFO, "raiduki");
    let _guard = span.enter();

    let cli = Cli::parse();

    match &cli.command {
        Commands::Ask(args) => {
            raiduki::semantic_search(&args.consulta, &pool).await?;
        }
        Commands::Load(args) => {
            let (data, embeddings) = raiduki::vectorize_csv(&args.origen).await?;
            if let Err(err)  = sqlx::query!("INSERT INTO datos_usuarios (data, embedding) SELECT * FROM UNNEST($1::text[], $2::vector[])", &data[..], &embeddings[..] as &[Vector]).execute(&pool).await {
                error!("{}",err.to_string());
                bail!(err);
            };
            info!("Se han cargado los datos exitosamente!");
        }
        Commands::Csv(args) => {
            todo!()
        }
    }
    Ok(())
}
