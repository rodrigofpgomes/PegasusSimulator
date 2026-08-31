import os
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator

# Defina a pasta onde estão os arquivos do TensorBoard ('tfevents')
LOG_DIR = "./logs/26-08-26_18-24-48-029776_SACDelayed_delay_2"
# Pasta onde você deseja salvar todos os arquivos CSV gerados
OUTPUT_FOLDER = "./download_csvs"

def download_all_graphs_individually(log_dir, output_folder):
    # Garante que a pasta de destino exista
    os.makedirs(output_folder, exist_ok=True)
    
    # Configura para carregar todos os dados (0 = sem limite)
    size_guidance = {event_accumulator.SCALARS: 0}
    ea = event_accumulator.EventAccumulator(log_dir, size_guidance=size_guidance)
    ea.Reload()
    
    # Pega a lista de todos os gráficos disponíveis
    scalar_tags = ea.Tags().get('scalars', [])
    
    if not scalar_tags:
        print("❌ Nenhum gráfico (scalar) encontrado na pasta informada.")
        return

    print(f"🔄 Encontrados {len(scalar_tags)} gráficos. Iniciando downloads automáticos...")
    
    # Loop automático por cada gráfico
    for tag in scalar_tags:
        events = ea.Scalars(tag)
        
        # Converte os dados do gráfico atual para um DataFrame
        df = pd.DataFrame([
            {
                "wall_time": e.wall_time, # Data/hora em formato timestamp do Unix
                "step": e.step,           # O número do passo/época
                "value": e.value          # O valor do gráfico naquele step
            } 
            for e in events
        ])
        
        # Corrige o nome caso o gráfico use barras (ex: 'train/loss' vira 'train_loss.csv')
        clean_tag_name = tag.replace("/", "_").replace("\\", "_")
        csv_filename = os.path.join(output_folder, f"{clean_tag_name}.csv")
        
        # Salva o arquivo CSV individual
        df.to_csv(csv_filename, index=False)
        print(f"📥 Baixado com sucesso: {csv_filename}")
        
    print(f"\n✨ Concluído! Todos os gráficos foram baixados na pasta '{output_folder}'.")

# Executar o download automatizado
download_all_graphs_individually(LOG_DIR, OUTPUT_FOLDER)