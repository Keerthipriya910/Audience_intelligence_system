from __future__ import annotations

import numpy as np


def explain_with_integrated_gradients(sentiment_engine, text: str, target_label: str | None = None, max_length: int = 128) -> list[dict]:
    """Compute token attributions for the loaded transformer using Captum LayerIntegratedGradients.

    This is intentionally on-demand because CPU Integrated Gradients is much more expensive
    than normal inference. It is used by the dashboard for publication-grade inspection of a
    selected comment instead of slowing every pipeline row.
    """
    if sentiment_engine.demo_mode:
        raise RuntimeError("Integrated Gradients requires Research Mode.")
    sentiment_engine._load()
    pipe=sentiment_engine._pipe
    model=pipe.model
    tokenizer=pipe.tokenizer
    model.eval()

    import torch
    from captum.attr import LayerIntegratedGradients

    encoded=tokenizer(text,return_tensors="pt",truncation=True,max_length=max_length)
    input_ids=encoded["input_ids"]
    attention_mask=encoded.get("attention_mask",torch.ones_like(input_ids))
    id2label={int(k):sentiment_engine._normalise_label(v) for k,v in model.config.id2label.items()}
    with torch.no_grad():
        logits=model(input_ids=input_ids,attention_mask=attention_mask).logits
    pred_id=int(torch.argmax(logits,dim=-1).item())
    if target_label is not None:
        candidates=[i for i,l in id2label.items() if l==target_label]
        target_id=candidates[0] if candidates else pred_id
    else:
        target_id=pred_id

    pad=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    baseline_ids=torch.full_like(input_ids,pad)
    # Preserve structural special tokens where possible.
    if tokenizer.cls_token_id is not None: baseline_ids[:,0]=tokenizer.cls_token_id
    if tokenizer.sep_token_id is not None:
        sep_positions=(input_ids==tokenizer.sep_token_id).nonzero(as_tuple=False)
        for b,pos in sep_positions.tolist(): baseline_ids[b,pos]=tokenizer.sep_token_id

    def forward(ids, mask):
        return model(input_ids=ids,attention_mask=mask).logits

    layer=getattr(getattr(model,"roberta",None),"embeddings",None)
    if layer is None:
        base=getattr(model,getattr(model,"base_model_prefix",""),None)
        layer=getattr(base,"embeddings",None)
    if layer is None:
        raise RuntimeError("Could not locate the transformer's embedding layer for Integrated Gradients.")

    lig=LayerIntegratedGradients(forward,layer)
    attrs=lig.attribute(inputs=input_ids,baselines=baseline_ids,additional_forward_args=(attention_mask,),target=target_id,n_steps=20)
    scores=attrs.sum(dim=-1).squeeze(0).detach().cpu().numpy()
    norm=float(np.linalg.norm(scores))
    if norm>0: scores=scores/norm
    tokens=tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).tolist())
    rows=[]
    specials=set(tokenizer.all_special_tokens)
    for token,score in zip(tokens,scores.tolist()):
        if token in specials: continue
        rows.append({"token":token.replace("▁"," ").strip() or token,"contribution":float(score)})
    rows.sort(key=lambda x:abs(x["contribution"]),reverse=True)
    return rows[:16]
