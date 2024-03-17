from mteb import AbsTask
import datasets

def unnest(ds) -> list[datasets.Dataset]:
    dss = []
    print("ds:",ds)
    for v in ds.values():
        if not isinstance(v, datasets.Dataset):
            dss.extend(unnest(v))
        else:
            dss.append(v)
    print("dss",dss)
    return dss

def combine_datasets(ds) -> datasets.Dataset:
    return datasets.concatenate_datasets([*unnest(ds)])

def classification(task: AbsTask) -> datasets.Dataset:
    ds = combine_datasets(task.dataset)
    return datasets.Dataset.from_dict({"text": ds["text"]})

def clustering(task: AbsTask) -> datasets.Dataset:
    ds = combine_datasets(task.dataset)
    sentences = [s for group in ds["sentences"] for s in group]
    return datasets.Dataset.from_dict({"text": sentences})

def pair_classification(task: AbsTask) -> datasets.Dataset:
    ds = combine_datasets(task.dataset)
    sent1 = [s for group in ds["sent1"] for s in group]
    sent2 = [s for group in ds["sent2"] for s in group]
    return datasets.Dataset.from_dict({"text": sent1 + sent2})

def reranking(task: AbsTask) -> datasets.Dataset:
    ds = combine_datasets(task.dataset)
    sentences = []
    if isinstance(ds[0]["query"], list):
        sentences.extend([s for group in ds["query"] for s in group])
    else:
        sentences.extend(ds["query"])
    
    sentences.extend([s for group in ds["positive"] for s in group])
    sentences.extend([s for group in ds["negative"] for s in group])

    return datasets.Dataset.from_dict({"text": sentences})

def retrieval(task: AbsTask) -> datasets.Dataset:

    s = task.description["eval_splits"][0]
    sentences = []
    if task.is_multilingual:
        for lang in task.langs:
            queries = [*task.queries[lang][s].values()]
            c = task.corpus[lang][s]
            if type(c.values()) == list:
                corpus = [f"{t['title']} {t['text']}".strip() for t in c]
            else:
                corpus = [f"{c['title'][i]} {c['text'][i]}".strip() for i in range(len(c['text']))]
            sentences.extend(queries + corpus)
    else:
        queries = [*task.queries[s].values()]
        c = task.corpus[s].values()
        corpus = [f"{t['title']} {t['text']}".strip() if "title" in t else t["text"].strip() for t in c]

        sentences.extend(queries + corpus)

    return datasets.Dataset.from_dict({"text": sentences})

def sts(task: AbsTask) -> datasets.Dataset:
    ds = combine_datasets(task.dataset)
    return datasets.Dataset.from_dict({"text": ds["sentence1"] + ds["sentence2"]})
    