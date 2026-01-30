import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from attacker import CCAD_Attack

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    local_image_path = "./img/cat.jpg"
    original_text = ["A photo of a cat staring forward."]

    try:
        img = Image.open(local_image_path).convert("RGB")
        initial_image_tensor = processor(images=img, return_tensors="pt").pixel_values.to(device)
    except:
        print("Image load failed, using dummy tensor.")
        initial_image_tensor = torch.zeros(1, 3, 224, 224).to(device)

    params = {
        'epsilon': 2/255.0, 
        'step_size': 0.5/255.0, 
        'num_iterations': 30, 
        'alpha': 0.2, 
        'beta': 0.2, 
        'gamma': 0.99
    }
    attacker = CCAD_Attack(model, processor, device=device, **params)

    final_img, final_txt, hist, best_record, init_metrics = attacker.run_attack(
        initial_image_tensor, original_text, "C"
    )

    i_loss, i_sim, i_ri, i_rt = init_metrics
    b_loss, b_sim, b_ri, b_rt = best_record['metrics']
    
    print("\n" + "="*75)
    print(f"{'CCAD PGD Attack Result':^75}")
    print("="*75)
    print(f"{'Metric':<22} | {'Initial':<20} | {'Best':<20}")
    print("-" * 75)
    print(f"{'Total Loss':<22} | {i_loss:<20.4f} | {b_loss:<20.4f}")
    print(f"{'Cosine Sim Loss':<22} | {i_sim:<20.4f} | {b_sim:<20.4f}")
    print("-" * 75)
    print(f"Initial Text: {original_text[0]}")
    print(f"Best Text:    {best_record['text']}")
    print("="*75)

if __name__ == '__main__':
    main()
