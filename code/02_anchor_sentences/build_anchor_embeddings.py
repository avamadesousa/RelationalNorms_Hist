"""
build_multi_anchors.py

Embeds 4 defintions per norm function (care, hierarchy,
transaction, mating), English and Chinese versions.

Saves to anchors_multi.npz:
  domain_names   (4,)              ["care","hierarchy","transaction","mating"]
  en_embs        (4, 4, 3072)       [domain, version(0-3), dim] float32, L2-normalized
  zh_embs        (4, 4, 3072)       same shape, Chinese-native anchors
  en_sentences   (4, 4)             text, for provenance
  zh_sentences   (4, 4)

"""

from openai import OpenAI
import numpy as np
import os

MODEL = "text-embedding-3-large"
OUT_PATH = "/home/local/PSYCH-ADS/avamadesousa/YES_lab/RelationalNormsNLP/anchors_multi.npz"

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

DOMAIN_NAMES = ["care", "hierarchy", "transaction", "mating"]

EN_SENTENCES = {
    "care": [
        "Giving or receiving support based on need, without creating a debt",
        "Protecting another person's well-being through attentive and reliable support",
        "Giving practical or emotional help without expecting repayment in return",
        "Providing dependable, unconditional assistance",
    ],
    "hierarchy": [
        "Coordinating behavior between people with unequal authority over one another",
        "Following established leadership roles when organizing shared behavior",
        "Organizing collective action according to rank, status, and responsibility",
        "Exercising greater decision-making authority over another person because of one's higher rank or position",
    ],
    "transaction": [
        "Balancing mutual benefits by keeping track of who owes what to whom",
        "Keeping cooperation fair through reciprocity, repayment, and balanced exchange",
        "Coordinating mutual gain by tracking contributions, obligations, and repayment",
        "Exchanging benefits in proportion to what each person contributes",
    ],
    "mating": [
        "Establishing and maintaining a sexual partnership",
        "Forming and sustaining a committed sexual relationship with another person",
        "Attracting and retaining a partner for an ongoing sexual relationship",
        "Creating and preserving a stable romantic and sexual partnership",
    ],
}

ZH_SENTENCES = {
    "care": [
        "根据需求给出或接受帮助和支持，不因此产生人情债",
        "通过关切的和值得信赖的支持，保护另一个人的福祉",
        "给予实践性的或情感性的支持，而不计较回报",
        "提供值得信赖的、不讲条件的协助",
    ],
    "hierarchy": [
        "在权威、地位不平等的人之间协调行动",
        "在有组织的协同行动中，遵从已确立的领导角色",
        "根据级别、地位的高低和职责的不同来组织集体行为",
        "因职级或职位较高，而对他人拥有更大的决策权。",
    ],
    "transaction": [
        "通过记录谁欠谁什么来平衡相互的利益",
        "通过互惠互利、偿付和对等的交换来维持合作的公平性",
        "通过记录各自的贡献、职责和偿付来协调相互的利益",
        "按照每人的贡献大小来分配利益",
    ],
    "mating": [
        "建立并维持性关系",
        "寻找并吸引一个可以发生性行为的性伴侣。",
        "与固定伴侣维持持续的性接触和肉体亲密关系。",
        "通过择偶和性关系来繁衍后代、延续血脉。",
    ],
}


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, 1e-9, None)


def embed_texts(texts):
    resp = client.embeddings.create(model=MODEL, input=texts)
    arr = np.array([d.embedding for d in resp.data], dtype=np.float32)
    return l2_normalize(arr)


def main():
    en_flat = [s for d in DOMAIN_NAMES for s in EN_SENTENCES[d]]
    zh_flat = [s for d in DOMAIN_NAMES for s in ZH_SENTENCES[d]]
    assert len(en_flat) == 16 and len(zh_flat) == 16

    print("[INIT] Embedding English anchor versions...")
    en_embs = embed_texts(en_flat).reshape(4, 4, -1)
    print("[INIT] Embedding Chinese anchor versions...")
    zh_embs = embed_texts(zh_flat).reshape(4, 4, -1)

    # Sanity check: how collinear are the 4 versions of the same domain vs. each other
    # (should be reasonably high, same concept) and vs. the other domains (should be lower)
    print("\nEnglish: version x version cosine within each domain")
    for i, d in enumerate(DOMAIN_NAMES):
        sim = en_embs[i] @ en_embs[i].T
        print(f"  {d}:\n{np.round(sim, 3)}")

    print("\nChinese: version x version cosine within each domain")
    for i, d in enumerate(DOMAIN_NAMES):
        sim = zh_embs[i] @ zh_embs[i].T
        print(f"  {d}:\n{np.round(sim, 3)}")

    en_sentences_arr = np.array([EN_SENTENCES[d] for d in DOMAIN_NAMES])
    zh_sentences_arr = np.array([ZH_SENTENCES[d] for d in DOMAIN_NAMES])

    np.savez(
        OUT_PATH,
        domain_names=np.array(DOMAIN_NAMES),
        en_embs=en_embs,
        zh_embs=zh_embs,
        en_sentences=en_sentences_arr,
        zh_sentences=zh_sentences_arr,
    )
    print(f"\n[DONE] Saved anchors to {OUT_PATH}")


if __name__ == "__main__":
    main()
 