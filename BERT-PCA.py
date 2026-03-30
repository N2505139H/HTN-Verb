import torch
from transformers import BertTokenizer, BertModel
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import warnings
from matplotlib import rcParams


warnings.filterwarnings("ignore", category=UserWarning, module='matplotlib')


rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertModel.from_pretrained('bert-base-chinese').to(device)



def get_bert_embeddings(texts, batch_size=8):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings.extend(outputs.last_hidden_state.mean(dim=1).cpu().numpy())  # Average Pooling
    return embeddings


def plot_pca_and_clusters(embeddings, labels, title="PCA降维结果"):
    pca = PCA(n_components=2, random_state=42)
    reduced_embeddings = pca.fit_transform(embeddings)


    clustering = AgglomerativeClustering(n_clusters=3)
    cluster_labels = clustering.fit_predict(reduced_embeddings)


    plt.figure(figsize=(8, 6))
    for i in range(3):
        indices = [j for j, label in enumerate(cluster_labels) if label == i]
        plt.scatter(reduced_embeddings[indices, 0], reduced_embeddings[indices, 1], label=f'Cluster {i + 1}', alpha=0.6)

    for i, label in enumerate(labels):
        plt.text(reduced_embeddings[i, 0], reduced_embeddings[i, 1], label, fontsize=9, alpha=0.7)

    plt.title(title)
    plt.xlabel("主成分 1")
    plt.ylabel("主成分 2")
    plt.legend()
    plt.show()



def main():

    task_file =  # Replace with the actual path to the task description file
    verb_file =  #  Replace with the actual path to the verb library file
    output_file =  # Output path


    from docx import Document
    doc = Document(task_file)
    task_descriptions = [para.text.strip() for para in doc.paragraphs if para.text.strip()]


    verb_df = pd.read_excel(verb_file, sheet_name=None)
    verb_dict = {sheet: verbs.iloc[:, 0].dropna().tolist() for sheet, verbs in verb_df.items()}


    print("开始计算任务描述嵌入...")
    task_embeddings = get_bert_embeddings(task_descriptions)
    print("任务描述嵌入完成。")


    print("开始计算动词库嵌入...")
    verb_embeddings = {}
    for sheet, verbs in verb_dict.items():
        print(f"处理动词分类：{sheet}，包含 {len(verbs)} 个动词。")
        verb_embeddings[sheet] = get_bert_embeddings(verbs)
    print("动词库嵌入完成。")


    with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:

        plot_pca_and_clusters(task_embeddings, task_descriptions, title="任务描述的PCA和聚类结果")
        task_pca = PCA(n_components=2, random_state=42).fit_transform(task_embeddings)
        task_df = pd.DataFrame({
            "任务描述": task_descriptions,
            "PCA 1": task_pca[:, 0],
            "PCA 2": task_pca[:, 1]
        })
        task_df.to_excel(writer, sheet_name="任务描述聚类", index=False)


        for sheet, embeddings in verb_embeddings.items():
            reduced_embeddings = PCA(n_components=2, random_state=42).fit_transform(embeddings)
            clustering = AgglomerativeClustering(n_clusters=3)
            cluster_labels = clustering.fit_predict(reduced_embeddings)

            verb_df = pd.DataFrame({
                "动词": verb_dict[sheet],
                "PCA 1": reduced_embeddings[:, 0],
                "PCA 2": reduced_embeddings[:, 1],
                "Cluster": cluster_labels
            })
            verb_df.to_excel(writer, sheet_name=sheet, index=False)
    print(f"聚类结果已保存到 {output_file}")



if __name__ == "__main__":
    main()