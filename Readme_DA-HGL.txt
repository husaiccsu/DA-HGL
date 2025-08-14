DA-HGL:
This repository contains the code and datasets for the DA-HGL method, as described in the manuscript.

Quick Start
1. Install Dependencies
   The code was developed and tested using python 3.10    
   pip install networkx
   pip install numpy
   pip install matplotlib
   pip install scikit-learn
   pip install joblib
   pip install seaborn

   
2 Download Data Files:
   Place the following files in the same directory as DA-HGL.py:
   [Species]_PPI2024.txt (e.g., Homo_sapiens_PPI2024.txt).
   [Species]_Domain2024.txt, [Species]_NCBI_Sequence.txt, [Species]_GO2024.txt,[Species]_GO2024__CAFA3.txt.
   Total_DAG.txt (GO hierarchy).
   Species: 'Homo_sapiens' 或 'Saccharomyces_cerevisiae'
   GOType: 'F'（MF）、'P'（BP）、'C'（CC）

3 Run Prediction:
  python DA-HGL.py
  Modify Species and GOType in DA-HG.py to switch between species (Human/Yeast) and ontologies (BP/MF/CC).
 

Custom Data Support
1. DA-HG.py: The DA-HG method
2. Homo_sapiens_Domain2024.txt: Protein Domain Profile for Human
3. Homo_sapiens__GO2024.txt: GO Annotation 2024 for Human
4. Homo_sapiens_GO2024_CAFA3.txt: The third Critical Assessment of Protein Function Annotation (CAFA3) dataset for Human
5. Homo_sapiens_PPI2024.txt: BioGRID PPI network for Human
6. Homo_sapiens_NCBI_Sequence.txt: The sequence data for Human
7. Saccharomyces_cerevisiae_Domain2024.txt: Protein Domain Profile for Yeast
8. Saccharomyces_cerevisiae_GO2024.txt: GO Annotation 2024 for Yeast
9. Saccharomyces_cerevisiae_GO2024_CAFA3.txt: The third Critical Assessment of Protein Function Annotation (CAFA3) dataset for Yeast
10. Saccharomyces_cerevisiae_PPI2024.txt: BioGRID PPI network for Yeast
11. Saccharomyces_cerevisiae_NCBI_Sequence.txt: The sequence data for Yeast
12. Total_DAG.txt: GO Ontology including Human and Yeast

License
  DA-HGL is released under the MIT License. Contact bihaizhao@163.com for questions.