use clap::Subcommand;
use clap::{Args, Parser};
use raiduki::{generar_embedded_csv, realizar_consulta};
use std::path::PathBuf;

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
    #[clap(help = "origen es el archivo a almacenar.")]
    #[arg(short, long)]
    origen: String,
    #[clap(help = "destino es el archivo en donde se almancenará el resultado.")]
    #[arg(short, long)]
    destino: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match &cli.command {
        Commands::Ask(args) => {
            realizar_consulta(&args.consulta).await?;
        }
        Commands::Load(args) => {
            generar_embedded_csv(
                &PathBuf::from(args.origen.clone()),
                &PathBuf::from(args.destino.clone()),
            )
            .await?;
        }
    }
    Ok(())
}
