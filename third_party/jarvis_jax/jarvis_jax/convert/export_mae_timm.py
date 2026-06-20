import argparse, numpy as np, timm, torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="vit_base_patch16_224.mae")
    args = ap.parse_args()
    try:
        m = timm.create_model(args.model, pretrained=True, num_classes=0)
    except Exception:
        args.model = "vit_base_patch16_224.augreg_in21k"   # supervised fallback
        m = timm.create_model(args.model, pretrained=True, num_classes=0)
    sd = {k: v.detach().cpu().numpy() for k, v in m.state_dict().items()}
    sd["__model_id__"] = np.array(args.model)
    np.savez(args.out, **sd)
    print(f"wrote {args.out} from {args.model} ({len(sd)} arrays)")

if __name__ == "__main__":
    main()
