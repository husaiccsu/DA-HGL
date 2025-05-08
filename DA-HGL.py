import argparse
import warnings
import math
import numpy as np
import pandas as pd
from math import factorial
from datetime import datetime
from collections import OrderedDict
from collections import Counter
from collections import defaultdict, deque
import random
import networkx as nx
from scipy.stats import pearsonr, spearmanr
import os
from functools import reduce
import scipy.spatial.distance as sd
from operator import itemgetter
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker  

from sklearn.linear_model import LogisticRegression
import sparse
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
from sklearn.decomposition import NMF
from scipy.linalg import svd  
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import non_negative_factorization
from scipy.stats import ks_2samp
from sklearn.model_selection import LeaveOneOut
import copy
import seaborn as sns
from scipy import stats
import plotly.express as px
from statsmodels.stats.multitest import multipletests

PPMatrix=[]
PGMatrix=np.empty_like([])
PDMatrix=np.empty_like([])
PCMatrix=[]
CGMatrix=[]
GPList=[]
GoList=[]
Domainlist=[]
Complexlist=[] 
proteinlist=[]
TrainProteinList=[]
TestProteinList=[]
NList=[]
N2List=[] 
CPList=[]
PCList=[]
PGList=[]
PDList=[]
PSList=[]
DPList=[]
PGONum=[]

α=0.1
β=0.1
ε=0.0001
γ=0.1
GOType='F'
CAFAStr=''
CAFAStr='_CAFA3'
IsTenFold=0
Species='Saccharomyces_cerevisiae'
#Species='Homo_sapiens'
warnings.filterwarnings('ignore', message='The parameter \'token_pattern\' will not be used since \'tokenizer\' is not None')


def ordered_unique(lst):
    seen = set()
    return [x for x in lst if not (x in seen or seen.add(x))]
    
def LoadPPI():
  global TrainProteinList
  global proteinlist
  global TestProteinList
  TempList=[]
  CAFAList=[]
  with open(Species+'_GO2024'+CAFAStr+'.txt', 'r') as file:
    for line in file:
      line = line.strip()
      beginstr,endstr,types=line.split('\t')
      if types!=GOType:
         continue      
      if beginstr not in TempList:
         TempList.append(beginstr)

  PList=[]
  with open(Species+'_PPI2024.txt', 'r') as file:
    for line in file:
      line = line.strip()
      beginstr,endstr=line.split('\t')
      if beginstr not in PList:
         PList.append(beginstr)
      if endstr not in PList:
         PList.append(endstr)        
    list1 = ordered_unique(PList) 
    list2 = ordered_unique(TempList)  
    proteinlist = [item for item in list1 if item in list2]
    listlen=len(proteinlist)
    file.seek(0)
    global PPMatrix
    PPMatrix = [[0 for _ in range(listlen)] for _ in range(listlen)]
    global NList
    NList=[[0 for j in range(0)] for i in range(listlen)]
    global N2List
    N2List=[[0 for j in range(0)] for i in range(listlen)]
    for line in file:
        line = line.strip()
        beginstr,endstr=line.split('\t')
        if (beginstr not in TempList) or (endstr not in TempList):
          continue
        Ipos=proteinlist.index(beginstr)
        JPos=proteinlist.index(endstr)
        PPMatrix[Ipos][JPos]=1
        PPMatrix[JPos][Ipos]=1  
        NList[Ipos].append(JPos)
        NList[JPos].append(Ipos)
        
    for i in range(0,listlen):
      for j in range(0,len(NList[i])):
         IPos=int(NList[i][j])
         for k in range(0,len(NList[IPos])):
            IPos2=int(NList[IPos][k])
            if (IPos2!=i) and (IPos2 not in N2List[i]):
              N2List[i].append(IPos2)

  with open(Species+'_GO2024'+CAFAStr+'.txt', 'r') as file:
    for line in file:
      line = line.strip()
      beginstr,endstr,types=line.split('\t')
      if types!=GOType:
         continue
      if beginstr not in proteinlist:
         continue 
      global GoList
      if endstr not in GoList:
         GoList.append(endstr)          
         
  global PGMatrix 
  PGMatrix=np.zeros((listlen,len(GoList)))
  global PGList
  PGList=[[0 for j in range(0)] for i in range(listlen)]
  global PGONum
  PGONum=[0 for i in range(listlen)] 
  global GPList
  GPList=[[0 for j in range(0)] for i in range(len(GoList))]
  with open(Species+'_GO2024'+CAFAStr+'.txt', 'r') as file:
   for line in file:
      line = line.strip()
      beginstr,endstr,types=line.split('\t')
      try:
          Ipos=proteinlist.index(beginstr)
      except ValueError:
          Ipos=-1
      if types!=GOType:
         continue    
      if Ipos==-1:
         continue     
      JPos=GoList.index(endstr)
      PGMatrix[Ipos][JPos]=1
      PGList[Ipos].append(JPos) 
      PGONum[Ipos]=PGONum[Ipos]+1 
      GPList[JPos].append(Ipos)        
      
def LoadMultiData():
   with open(Species+'_Domain2024.txt', 'r') as file:
    for line in file:
        line = line.strip()
        beginstr,endstr=line.split('\t')
        global Domainlist          
        if endstr not in Domainlist:
          Domainlist.append(endstr)      
    PSize=len(proteinlist)
    listlen=len(Domainlist)
    global PDMatrix
    global DPList
    PDMatrix=np.zeros((PSize,listlen))
    DPList=[[0 for j in range(0)] for i in range(len(Domainlist))]  
    
    file.seek(0)
    for line in file:
        line = line.strip()
        beginstr,endstr=line.split('\t')
        try:
          Ipos=proteinlist.index(beginstr)
        except ValueError:
          Ipos=-1
        if Ipos>=0:
          JPos=Domainlist.index(endstr)
          PDMatrix[Ipos][JPos]=1
          DPList[JPos].append(Ipos) 
      
    
def calculate_aac(sequence):
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    count = Counter(sequence)
    total_count = len(sequence)
    aac_vector = [count[aa] / total_count for aa in amino_acids]
    return aac_vector

def calculate_pse_aac(sequence, lambda_value=20, w=0.05):
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    count = Counter(sequence)
    total_count = len(sequence)
    aac_vector = calculate_aac(sequence)
    autocorr_vectors = []
    for lag in range(1, lambda_value + 1):
        corr_vector = []
        for i in range(total_count - lag):
            if sequence[i] in amino_acids and sequence[i + lag] in amino_acids:
                corr = 1 if sequence[i] == sequence[i + lag] else -1
                corr_vector.append(corr)
        autocorr = np.mean(corr_vector) if corr_vector else 0
        autocorr_vectors.append(autocorr)
    pse_aac_vector = aac_vector + [w * autocorr for autocorr in autocorr_vectors]
    norm_factor = sum(pse_aac_vector)
    pse_aac_vector = [value / norm_factor for value in pse_aac_vector]

    return pse_aac_vector

def calculate_similarity(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)
    vec1 =np.array(vector1)
    vec2 =np.array(vector2)
    cos_similarity = dot_product / (norm1 * norm2) if norm1 != 0 and norm2 != 0 else 0 #余弦距离
    return cos_similarity

def read_sequences(file_path):
    sequences = {}
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                parts = line.split('\t')         
                if parts[0] not in proteinlist:
                  continue 
                if len(parts) == 2:
                    protein_id = parts[0]
                    sequence = parts[1]
                    sequences[protein_id] = sequence
    return sequences

def GetSequenceSimilarity():
    file_path = Species+'_NCBI_Sequence.txt'  
    sequences = read_sequences(file_path)
    protein_ids = list(sequences.keys())
    similarity_matrix = np.zeros((len(proteinlist), len(proteinlist)))
    for i in range(len(protein_ids)-1):
        print(i+1,'/',len(protein_ids)-1,end='\r')
        id1 = protein_ids[i]
        if id1 not in proteinlist:
            break
        Pos1=proteinlist.index(id1)    
        seq1 = sequences[id1]       
        vector1 = calculate_pse_aac(seq1)
        for j in range(i, len(protein_ids)):
            id2 = protein_ids[j]
            if id2 not in proteinlist:
               break
            Pos2=proteinlist.index(id2)
            if PPMatrix[Pos1][Pos2]==1:
              seq2 = sequences[id2]
              vector2 = calculate_pse_aac(seq2)
              sim1 = calculate_similarity(vector1, vector2)
              similarity_matrix[Pos1][Pos2]=sim1
              similarity_matrix[Pos2][Pos1]=sim1
    return similarity_matrix
    
def Weight_Protein():
   listlen=len(proteinlist)
   Seq_Sim=np.zeros((listlen,listlen))
   Seq_Sim=GetSequenceSimilarity()       
   min_val = Seq_Sim.min()
   max_val = Seq_Sim.max()
   WeightP = (Seq_Sim - min_val) / (max_val - min_val)
   return WeightP

def Weight_Domain2():
    listlen=len(proteinlist)
    global  PDList 
    N_CPList=[[0 for j in range(0)] for i in range(listlen)]
    PDList=[[0 for j in range(0)] for i in range(listlen)]
    PDList2=[[0 for j in range(0)] for i in range(listlen)] 
  
    for i in range(0,listlen): 
     for j in range(0,len(Domainlist)):
       if PDMatrix[i][j]==1:
          PDList[i].append(j)
       for k in range(len(NList[i])):
          if PDMatrix[int(NList[i][k])][j]==1:
            PDList2[i].append(j) 
            break  
    domain_docs = []
    domain_docs2 = []
    for i in range(len(proteinlist)):
        domains = [Domainlist[j] for j in PDList[i]]  
        domain_docs.append(" ".join(domains))
        domains2 = [Domainlist[j] for j in PDList2[i]]  
        domain_docs2.append(" ".join(domains2))
    
    vectorizer = TfidfVectorizer(tokenizer=lambda x: x.split(), lowercase=False)
    tfidf_matrix = vectorizer.fit_transform(domain_docs)
    vectorizer2 = TfidfVectorizer(tokenizer=lambda x: x.split(), lowercase=False)
    tfidf_matrix2 = vectorizer2.fit_transform(domain_docs2)   
    b=0.1
    similarity_matrix = cosine_similarity(tfidf_matrix)*(1-b)+cosine_similarity(tfidf_matrix2)*b   
    similarity_matrix = (similarity_matrix - similarity_matrix.min()) / (similarity_matrix.max() - similarity_matrix.min())   
    return similarity_matrix

def train_model(PGMatrix, M, GO_DAG_Matrix,alpha, beta):
    max_iter=200
    lr=0.001
    lambda_=0.001
    PG = np.copy(PGMatrix).astype(np.float64)  
    n_proteins, n_GO = PG.shape
    if n_GO-1<500:
      k=n_GO-1
    else:
      k=500        
    U, s, Vt = svd(PG, full_matrices=False)
    U = U[:, :k] * np.sqrt(s[:k])
    V = Vt[:k, :].T * np.sqrt(s[:k])
    L_p = compute_laplacian(M)
    L_go = compute_laplacian(GO_DAG_Matrix)
    tol = 1e-6  
    prev_loss = np.inf 
    best_loss = float('inf')
    #---------------------------------------------------------------------------------
    for epoch in range(1,max_iter+1):        
        reconstruction_error = PG - U @ V.T   
        current_loss = np.mean(reconstruction_error ** 2) 
        if abs(current_loss - prev_loss) < tol:
           break
        prev_loss = current_loss         
        scale_factor = 1.0 / (n_proteins * n_GO)
        grad_U = -2 * scale_factor * reconstruction_error @ V + 2*lambda_*U + 2*alpha * L_p @ U
        grad_V = -2 * scale_factor * reconstruction_error.T @ U + 2*lambda_*V + 2*beta * L_go @ V
        U -= lr * grad_U
        V -= lr * grad_V           
    return U, V,epoch
    
def MVFP(M,alpha, beta,train_index, test_index):
    #------------------------------------------------------------------------------------
    n_GO=len(GoList) 
    GO_DAG_Matrix = np.zeros((n_GO, n_GO))
    G=nx.DiGraph()
    with open('Total_DAG.txt', 'r') as file:  
     for line in file:
      line = line.strip()
      beginstr,endstr,types,relation=line.split('\t')
      if types!=GOType:
         continue       
      if  relation=='is_a':        
        G.add_edge(beginstr,endstr,weight=0.4)
      else:
        G.add_edge(beginstr,endstr,weight=0.3)       
       
    for i in range(0,n_GO):    
      print(i+1,'/',n_GO,end='\r')
      for j in range(0,n_GO):
        if i==j:
          GO_DAG_Matrix[i][j]=1
          continue
        if GoList[i] in G.nodes() and GoList[j] in G.nodes():
            sp=max_path_product(G, GoList[i],GoList[j])
            GO_DAG_Matrix[i][j]=sp 
    #-----------------------------------------------------------------------------------
    LP=len(proteinlist)
    LG=len(GoList)
    PNeighborList=[[0 for j in range(0)] for i in range(LP)]
    GNeighborList=[[0 for j in range(0)] for i in range(LG)]
    HengVector=np.zeros(LP)
    ShuVector=np.zeros(LP)
    for i in range(LP):
     for j in range(LP):
       if M[i][j]!=0 and j!=i:
         PNeighborList[i].append(j)
    for i in range(LG):
     for j in range(LG):
       if GO_DAG_Matrix[i][j]!=0:
         GNeighborList[i].append(j)   
    for i in range(LP):  
     sumi=0
     sumj=0
     for j in range(LP):
        sumi+=M[i][j]
        sumj+=M[j][i]
     HengVector[i]=sumi
     ShuVector[i]=sumj  
    for i in range(LP):
      print(i+1,'/',LP,end='\r')
      for j in range(LP):
        sumi=HengVector[i]
        sumj=ShuVector[j]
        if sumi*sumj!=0:
          M[i][j]=M[i][j]/math.sqrt(sumi*sumj)
        else:
          M[i][j]=0   
    for i in range(LG):
      print(i+1,'/',LG,end='\r')
      for j in range(LG):
        sumi=0
        sumj=0
        for k in range(LG):
          sumi+=GO_DAG_Matrix[i][k]
          sumj+=GO_DAG_Matrix[k][j]
        if sumi*sumj!=0:
          GO_DAG_Matrix[i][j]=GO_DAG_Matrix[i][j]/math.sqrt(sumi*sumj)
        else:
          GO_DAG_Matrix[i][j]=0
    #-----------------------------------------------------------------------------------
    RunningCount=LP
    #RunningCount=10
    AVGN=sum(PGONum)/len(PGONum)
    AVGValue=math.ceil(AVGN)
    def process_one_protein(i):    
       if PGONum[i]==0:
         return
       if len(train_index)>0 and (i not in test_index): 
         return None  
       PG=np.copy(PGMatrix)
       if len(train_index)>0:      
          PG[test_index, :] = 0
       else:
          PG[i, :] = 0
       U, V,t= train_model(PG, M, GO_DAG_Matrix,alpha, beta)
       predicted_scores=np.zeros(LG)
       predicted_scores = U[i, :] @ V.T 
       return predicted_scores
    results = Parallel(n_jobs=3)(delayed(process_one_protein)(i) for i in range(RunningCount))
    all_predictions =np.zeros((LP,LG))
    for i, predicted_scores in enumerate(results):
      if predicted_scores is not None:
        for j in range (len(predicted_scores)):
          all_predictions[i, j] = predicted_scores[j]
    if len(train_index)>0:
       return all_predictions[test_index, :]
    else:
       return all_predictions 
       


def compute_laplacian(M):
    D = np.diag(M.sum(axis=1))
    return D - M
  
def max_path_product(G, source, target):
    all_paths = list(nx.all_simple_paths(G, source, target))
    if not all_paths:
        return 0    
    products = [reduce(lambda x, y: x * y, [G[u][v]['weight'] for u, v in zip(path, path[1:])], 1) for path in all_paths]
    max_product=max(products)
    return max_product 


    
def evaluate_predictions(true_labels, predictions):
    pred_z = (predictions - np.mean(predictions)) / np.std(predictions)
    pred_normalized = (pred_z - np.min(pred_z)) / (np.max(pred_z) - np.min(pred_z))   
    sorted_indices = np.argsort(-pred_normalized, axis=1)
    sorted_scores = np.array([pred_normalized[i][indices] for i, indices in enumerate(sorted_indices)])
    sorted_labels = np.array([true_labels[i][indices] for i, indices in enumerate(sorted_indices)])   
    best_fmax = 0.0
    for threshold in np.linspace(0, 1, 1000)[::-1]:
        binary_preds = (sorted_scores >= threshold).astype(int)
        tp = np.sum((binary_preds == 1) & (sorted_labels == 1))
        fp = np.sum((binary_preds == 1) & (sorted_labels == 0))
        fn = np.sum((binary_preds == 0) & (sorted_labels == 1))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        best_fmax = max(best_fmax, f1)
    y_true = true_labels.flatten()
    y_score = pred_normalized.flatten()
    auroc = roc_auc_score(y_true, y_score)
    aupr = average_precision_score(y_true, y_score)
    
    return best_fmax, auroc, aupr

def get_metrics_and_curves(true_labels, predictions):
    fmax, auroc, aupr = evaluate_predictions(true_labels, predictions)
    y_true = true_labels.flatten()
    y_score = predictions.flatten()
    fpr, tpr, _ = roc_curve(y_true, y_score)
    precision, recall, _ = precision_recall_curve(y_true, y_score)    
    return {
        "fmax": fmax,
        "auroc": auroc,
        "aupr": aupr,
        "roc_curve": (fpr, tpr),
        "pr_curve": (precision, recall)
    }

def plot_curves_with_metrics(method_results, true_labels):
    plt.figure(figsize=(6, 5))
    
    metrics = {}
    for name, pred in method_results.items():
        result = get_metrics_and_curves(true_labels, pred)
        metrics[name] = {
            "Fmax": result["fmax"],
            "AUROC": result["auroc"],
            "AUPR": result["aupr"]
        }
        fpr, tpr = result["roc_curve"]
        plt.plot(fpr, tpr, label=f'{name} (AUC={result["auroc"]:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc='lower right')
    plt.tight_layout()
    if IsTenFold==0:
       FoldStr='_one'
    else:
       FoldStr='_ten' 
    output_file = Species+'_'+GOType+CAFAStr+FoldStr+'_ROC.jpg'
    plt.savefig(output_file, dpi=720)
    plt.close()
    plt.figure(figsize=(6, 5))
    for name, pred in method_results.items():
        result = get_metrics_and_curves(true_labels, pred)
        precision, recall = result["pr_curve"]
        plt.plot(recall, precision, label=f'{name} (AUPR={result["aupr"]:.3f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR Curves')
    plt.legend(loc='upper right')

    plt.tight_layout()

    if IsTenFold==0:
       FoldStr='_one'
    else:
       FoldStr='_ten'
    output_file = Species+'_'+GOType+CAFAStr+FoldStr+'_PR.jpg'
    plt.savefig(output_file, dpi=720)
    plt.close()
    print("\nMetrics Summary:")
    print("{:<15} {:<10} {:<10} {:<10}".format("Method", "Fmax", "AUROC", "AUPR"))
    for name, data in metrics.items():
        print("{:<15} {:<10.3f} {:<10.3f} {:<10.3f}".format(
            name, data["Fmax"], data["AUROC"], data["AUPR"]))

def single_evaluate(methods,true_labels, predictions):
    fmax, auroc, aupr = evaluate_predictions(true_labels, predictions)
    print("{:<15} {:<10.3f} {:<10.3f} {:<10.3f}".format(methods, fmax, auroc, aupr))

def ten_fold_cross_validation():
    global IsTenFold
    IsTenFold=1
    print("Loading Data:"+Species+'_'+GOType+CAFAStr, datetime.now().strftime('%Y-%m-%d %H:%M:%S')) 
    LoadPPI()    
    LoadMultiData()
    method_results = {}
    kf = KFold(n_splits=10, shuffle=True, random_state=42)  
    splits = list(kf.split(proteinlist))
    all_predictions_MVFP = np.zeros((len(proteinlist), len(GoList)))
    WeightP=Weight_Protein()
    WeightD=Weight_Domain2()
    M1 = WeightP.astype(np.float32)
    M3 = WeightD.astype(np.float32)
    if GOType=='P':
        a=0.5
    elif GOType=='F':         
        a=0.3
    else:
        a=0.6
    M=a*M1+(1-a)*M3
    for fold_idx, (global_train_index, global_test_index) in enumerate(splits):        
        PG=np.copy(PGMatrix)
        tenfold_labels=PG[global_test_index, : ] 
        predictions_MVFP=MVFP(M,0.1,0.1,global_train_index, global_test_index)
        all_predictions_MVFP[global_test_index, :]=predictions_MVFP
        single_evaluate('DA-HGL',tenfold_labels,predictions_MVFP)
    method_results['DA-HGL'] = all_predictions_MVFP
    plot_curves_with_metrics(method_results, PGMatrix)
    
def leave_one_out_validation():
    global IsTenFold
    IsTenFold=0
    print("Loading Data:"+Species+'_'+GOType+CAFAStr, datetime.now().strftime('%Y-%m-%d %H:%M:%S')) 
    LoadPPI()    
    LoadMultiData()        
    method_results = {}      
    WeightP=Weight_Protein()
    WeightD=Weight_Domain2()
    M1 = WeightP.astype(np.float32)
    M3 = WeightD.astype(np.float32)   
    if GOType=='P':
        a=0.5
    elif GOType=='F':         
        a=0.3
    else:
        a=0.6
    M=a*M1+(1-a)*M3
    all_predictions1=MVFP(M,0.1,0.1,[],[]) 
    method_results['DA-HGL'] = all_predictions1
    plot_curves_with_metrics(method_results, PGMatrix)



def main():
   leave_one_out_validation()
if __name__ == "__main__":
	main()
   