from ultralytics import YOLO

# https://docs.ultralytics.com/pt/tasks/classify

versao_YOLO = "yolo11n-cls.pt"
model = YOLO(versao_YOLO)
file = "data\dataset_yolo_v1"


if __name__ == "__main__":
    model.train(data=file, 
                epochs=50,

                project='runs/classify_v1',
                exist_ok=True,
                resume=False,
                )
    # results = model.train(        
    #     data=file,
        
    #     # Configurações SEGURAS para evitar crash
    #     workers=6,       # Muito menos workers
    #     device=[0],      # GPU única
    #     batch=12,         # Batch muito pequeno para economizar VRAM
    #     imgsz=512,       # Imagem menor = menos processamento
        
    #     # Configurações de treinamento conservadoras
    #     epochs=300,      # Menos épocas inicialmente
    #     patience=200,     # Paciência menor
    #     optimizer='SGD', # SGD usa menos memória que AdamW
        
    #     # SEM data augmentation pesada (economiza processamento)
    #     mosaic=0.7,      # Reduz mosaic
    #     mixup=0.0,       # Remove mixup
    #     copy_paste=0.0,  # Remove copy-paste
        
    #     # Configurações de projeto
    #     project='runs/modelv5/yolo-safe',
    #     name='300-Epochs',
    #     exist_ok=True,
    #     resume=False,
        
    #     # Configurações mínimas
    #     cos_lr=False,    # Remove cosine scheduler
    #     warmup_epochs=1, # Warm-up mínimo
    #     weight_decay=0,
        
    #     # Salvamento
    #     save_period=25,  # Salva menos frequentemente
    #     val=True,
    #     plots=False,     # Remove plots para economizar recursos
    #     verbose=True
    #     )

