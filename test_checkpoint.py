import argparse
import os
import pickle as pk
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from dataloader import IEMOCAPDataset, MELDDataset
from model import MaskedNLLLoss, MaskedKLDivLoss, Transformer_Based_Model

LABEL2NAME = {
    0: "neutral",
    1: "surprise",
    2: "fear",
    3: "sadness",
    4: "joy",
    5: "disgust",
    6: "anger",
}

DEFAULT_SAMPLE_SENTENCES = [
    "I am really happy with this result!",
    "I am disappointed and sad about what happened.",
    "Wait, what? I did not expect this.",
    "This makes me so angry right now.",
]


def infer_meld_dims(path):
    with open(path, "rb") as f:
        data = pk.load(f)
    video_text = data[3]
    video_audio = data[7]
    video_visual = data[8]
    train_vid = data[10]
    test_vid = data[11]
    sample_vid = next(iter(train_vid)) if len(train_vid) > 0 else next(iter(test_vid))
    text_dim = int(np.asarray(video_text[sample_vid]).shape[-1])
    audio_dim = int(np.asarray(video_audio[sample_vid]).shape[-1])
    visual_dim = int(np.asarray(video_visual[sample_vid]).shape[-1])
    return text_dim, visual_dim, audio_dim


def build_loader(dataset_name, meld_pkl_path, batch_size):
    if dataset_name == "MELD":
        testset = MELDDataset(meld_pkl_path, train=False)
    else:
        testset = IEMOCAPDataset(train=False)
    return DataLoader(
        testset,
        batch_size=batch_size,
        collate_fn=testset.collate_fn,
        num_workers=0,
        pin_memory=False,
    )


def evaluate(model, loss_function, kl_loss, dataloader, device):
    model.eval()
    losses, preds, labels, masks = [], [], [], []

    with torch.no_grad():
        for data in dataloader:
            textf, visuf, acouf, qmask, umask, label = [d.to(device) for d in data[:-1]]
            qmask = qmask.permute(1, 0, 2)
            lengths = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(len(umask))]

            log_prob1, log_prob2, log_prob3, all_log_prob, all_prob, \
            kl_log_prob1, kl_log_prob2, kl_log_prob3, kl_all_prob = model(textf, visuf, acouf, umask, qmask, lengths)

            lp_1 = log_prob1.view(-1, log_prob1.size()[2])
            lp_2 = log_prob2.view(-1, log_prob2.size()[2])
            lp_3 = log_prob3.view(-1, log_prob3.size()[2])
            lp_all = all_log_prob.view(-1, all_log_prob.size()[2])
            labels_ = label.view(-1)

            kl_lp_1 = kl_log_prob1.view(-1, kl_log_prob1.size()[2])
            kl_lp_2 = kl_log_prob2.view(-1, kl_log_prob2.size()[2])
            kl_lp_3 = kl_log_prob3.view(-1, kl_log_prob3.size()[2])
            kl_p_all = kl_all_prob.view(-1, kl_all_prob.size()[2])

            loss = (
                loss_function(lp_all, labels_, umask)
                + (loss_function(lp_1, labels_, umask) + loss_function(lp_2, labels_, umask) + loss_function(lp_3, labels_, umask))
                + (kl_loss(kl_lp_1, kl_p_all, umask) + kl_loss(kl_lp_2, kl_p_all, umask) + kl_loss(kl_lp_3, kl_p_all, umask))
            )

            pred_ = torch.argmax(all_prob.view(-1, all_prob.size()[2]), dim=1)

            preds.append(pred_.cpu().numpy())
            labels.append(labels_.cpu().numpy())
            masks.append(umask.view(-1).cpu().numpy())
            losses.append(loss.item() * masks[-1].sum())

    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    masks = np.concatenate(masks)
    avg_loss = round(np.sum(losses) / np.sum(masks), 4)
    avg_acc = round(accuracy_score(labels, preds, sample_weight=masks) * 100, 2)
    avg_f1 = round(f1_score(labels, preds, sample_weight=masks, average="weighted") * 100, 2)
    return avg_loss, avg_acc, avg_f1, labels, preds, masks


def read_sentences_from_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines() if line.strip()]


def predict_sentences(text_checkpoint, sentences, device, max_length=128):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    tokenizer = AutoTokenizer.from_pretrained(text_checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(text_checkpoint).to(device)
    model.eval()

    encoded = tokenizer(
        sentences,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        logits = model(**encoded).logits
        probs = torch.softmax(logits, dim=-1)
        confs, preds = torch.max(probs, dim=1)

    print("")
    print("Sentence predictions ({})".format(text_checkpoint))
    for sent, pred, conf in zip(sentences, preds.cpu().tolist(), confs.cpu().tolist()):
        label_name = LABEL2NAME.get(int(pred), str(int(pred)))
        print("- text: {}".format(sent))
        print("  pred: {} ({}) | confidence: {:.4f}".format(int(pred), label_name, float(conf)))


def main():
    parser = argparse.ArgumentParser(description="Load a train.py checkpoint and evaluate it on the test split.")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint saved by train.py")
    parser.add_argument("--dataset", default="", choices=["", "MELD", "IEMOCAP"], help="Optional override of dataset")
    parser.add_argument("--meld-pkl-path", default="", help="MELD pickle path override")
    parser.add_argument("--batch-size", type=int, default=0, help="Batch size override; 0 uses checkpoint args")
    parser.add_argument("--no-cuda", action="store_true", default=False, help="Force CPU evaluation")
    parser.add_argument("--skip-test-eval", action="store_true", default=False, help="Skip SDT test split evaluation")
    parser.add_argument("--text-checkpoint", default="", help="HF text-classifier checkpoint dir for sentence prediction")
    parser.add_argument("--predict-sentences", nargs="*", default=[], help="Inline sentences for prediction")
    parser.add_argument("--predict-sentences-file", default="", help="Text file path (one sentence per line)")
    parser.add_argument("--predict-sample-sentences", action="store_true", default=False, help="Use built-in sample sentences")
    parser.add_argument("--max-length", type=int, default=128, help="Tokenizer max_length for sentence prediction")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    if not args.skip_test_eval:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        ckpt_args = checkpoint.get("args", {})

        dataset_name = args.dataset if args.dataset else ckpt_args.get("Dataset", "MELD")
        batch_size = args.batch_size if args.batch_size > 0 else int(ckpt_args.get("batch_size", 8))
        meld_pkl_path = args.meld_pkl_path if args.meld_pkl_path else ckpt_args.get("meld_pkl_path", "data/meld_multimodal_features.pkl")
        if dataset_name == "MELD" and not os.path.exists(meld_pkl_path):
            raise FileNotFoundError("MELD pickle not found: {}".format(meld_pkl_path))

        state_dict = checkpoint["model_state_dict"]
        ckpt_d_text = int(state_dict["textf_input.weight"].shape[1])
        ckpt_d_visual = int(state_dict["visuf_input.weight"].shape[1])
        ckpt_d_audio = int(state_dict["acouf_input.weight"].shape[1])

        if dataset_name == "MELD":
            pkl_d_text, pkl_d_visual, pkl_d_audio = infer_meld_dims(meld_pkl_path)
            if (pkl_d_text, pkl_d_visual, pkl_d_audio) != (ckpt_d_text, ckpt_d_visual, ckpt_d_audio):
                raise RuntimeError(
                    "Feature dimension mismatch between checkpoint and MELD pickle. "
                    "checkpoint(text/visual/audio)=({}/{}/{}), pickle=({}/{}/{}). "
                    "Use the same --meld-pkl-path that was used during training (checkpoint args meld_pkl_path: {}).".format(
                        ckpt_d_text, ckpt_d_visual, ckpt_d_audio,
                        pkl_d_text, pkl_d_visual, pkl_d_audio,
                        ckpt_args.get("meld_pkl_path", "N/A"),
                    )
                )

        d_text, d_visual, d_audio = ckpt_d_text, ckpt_d_visual, ckpt_d_audio

        n_classes = 7 if dataset_name == "MELD" else 6
        n_speakers = 9 if dataset_name == "MELD" else 2

        model = Transformer_Based_Model(
            dataset_name,
            int(ckpt_args.get("temp", 1)),
            d_text,
            d_visual,
            d_audio,
            int(ckpt_args.get("n_head", 8)),
            n_classes=n_classes,
            hidden_dim=int(ckpt_args.get("hidden_dim", 1024)),
            n_speakers=n_speakers,
            dropout=float(ckpt_args.get("dropout", 0.5)),
        ).to(device)
        model.load_state_dict(state_dict, strict=False)

        if dataset_name == "IEMOCAP":
            loss_weights = torch.FloatTensor([1 / 0.086747, 1 / 0.144406, 1 / 0.227883, 1 / 0.160585, 1 / 0.127711, 1 / 0.252668]).to(device)
            loss_function = MaskedNLLLoss(loss_weights)
        else:
            loss_function = MaskedNLLLoss()
        kl_loss = MaskedKLDivLoss()

        test_loader = build_loader(dataset_name, meld_pkl_path, batch_size=batch_size)
        test_loss, test_acc, test_f1, labels, preds, masks = evaluate(model, loss_function, kl_loss, test_loader, device)

        print("Checkpoint :", args.checkpoint)
        print("Dataset    :", dataset_name)
        print("Device     :", device)
        print("Batch size :", batch_size)
        if dataset_name == "MELD":
            print("MELD pkl   :", meld_pkl_path)
        print("Test loss  :", test_loss)
        print("Test acc   :", test_acc)
        print("Test f1    :", test_f1)
        print("")
        print(classification_report(labels, preds, sample_weight=masks, digits=4, zero_division=0))
        print(confusion_matrix(labels, preds, sample_weight=masks))

    prediction_sentences = []
    if args.predict_sample_sentences:
        prediction_sentences.extend(DEFAULT_SAMPLE_SENTENCES)
    if args.predict_sentences_file:
        prediction_sentences.extend(read_sentences_from_file(args.predict_sentences_file))
    if args.predict_sentences:
        prediction_sentences.extend(args.predict_sentences)

    if prediction_sentences:
        text_checkpoint = args.text_checkpoint
        if not text_checkpoint:
            raise ValueError("For sentence prediction, provide --text-checkpoint pointing to a HF text-classifier checkpoint directory.")
        predict_sentences(text_checkpoint, prediction_sentences, device, max_length=args.max_length)


if __name__ == "__main__":
    main()
