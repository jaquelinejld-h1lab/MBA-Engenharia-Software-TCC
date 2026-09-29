# Aprendizado de Máquina Aplicado à Predição de Doenças Crônicas: 
# Um Estudo de Caso de Hipertensão Arterial

# Conjunto de bibliotecas e funções utilizadas
# Manipulação dos dados
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import scipy.stats as ss
import math
from scipy.stats import uniform

# Pré-processamento dos dados
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

# Imputação
from sklearn.impute import KNNImputer

# Balanceamento
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import SMOTENC
from imblearn.under_sampling import NearMiss


# Divisão dos dados
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score, RandomizedSearchCV, ParameterSampler

# Seleção de Variáveis
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier

# Algoritmos
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# Métricas para avaliação do modelo
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc, roc_auc_score, roc_curve, precision_score, recall_score, f1_score, make_scorer

# Otimizar hiperparametros
import optuna
from optuna.samplers import TPESampler
import time
from imblearn.pipeline import Pipeline

# Interpretabilidade
import shap

# Executar e atualizar comandos desse arquivo no jupyter
import locale
from tqdm import tqdm #barra progresso select var

# Evitar avisos
import warnings
warnings.filterwarnings('ignore')

# Suprimir avisos
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
#warnings.filterwarnings('ignore', category=ConvergenceWarning)


# Configure o pandas para exibir todas as colunas
pd.set_option('display.max_columns', None)

# Ajustando o estilo do seaborn para melhorar a aparência do gráfico
sns.set(style="whitegrid")

# ------------------------------------------------------------------ #
# Função para carregar os dados, fazer ajustes na população de interesse e dividir os dados

def carregar_dividir_dados(path, teste):
    # Lista de variáveis a serem carregadas (em maiúsculas)
    variaveis = [
        'Z051', 'P005', 'Q001', 'Q002', 'Q003', 'Q004', 'Q005', 'Q006', 'Q01801', 'Q01802', 
        'Q01803', 'Q01804', 'Q01805', 'Q01806', 'Q01807', 'Q028', 'Q124', 'W00407', 'W00408', 
        'Z002', 'Z001', 'Z003', 'REGIAO', 'C001', 'C010', 'C011', 'E001', 'E01602', 'E01604', 
        'E01802', 'E01804', 'F00102', 'F00702', 'F00802', 'I001', 'J002', 'Z004', 'Z005', 'J037', 
        'N001', 'N004', 'N005', 'N010', 'N011', 'W00303', 'Z025', 'Z026', 'Z027', 'Z031', 'Z032', 
        'Z033', 'Z036', 'Z035', 'Z044', 'Z045', 'Z049', 'P002', 'P006', 'P007', 'P009', 'P011', 
        'P012', 'P013', 'P014', 'P015', 'P016', 'P018', 'P020', 'P021', 'P023', 'P024', 'P025', 
        'P026', 'P02601', 'P027', 'P034', 'P035', 'P036','P038', 'P039', 'P03901', 'P03902', 'P03903', 
        'P040', 'P04101', 'P04102', 'P042', 'P04301', 'P04302', 'P044', 'P04401', 'P04403', 
        'P04404', 'P045', 'P046', 'P050', 'P051', 'P052', 'P053', 'P068', 'Q030', 'Q060', 'Q063', 
        'Q068', 'Q132'
    ]
    
    # Dicionário para renomear colunas
    colunas_renomeadas = {
        'Z051': 'questionario', 'P005': 'gravida', 'Q001': 'pa_medida', 'Q002': 'diag_ha',
        'Q003': 'idade_diag_ha', 'Q004': 'freq_serv_sau', 'Q005': 'mot_vfreq_serv_sau',
        'Q006': 'remed_ha', 'Q01801': 'med_reco_almt_saud', 'Q01802': 'med_reco_mtr_peso',
        'Q01803': 'med_reco_menos_sal', 'Q01804': 'med_reco_prat_atf', 'Q01805': 'med_reco_nfumar',
        'Q01806': 'med_reco_beber_menos', 'Q01807': 'med_reco_acmp_reg', 'Q028': 'ha_limita',
        'Q124': 'diag_renalc', 'W00407': 'pa_sistolica', 'W00408': 'pa_distolica', 'Z002': 'idade',
        'Z001': 'sexo', 'Z003': 'etnia', 'REGIAO': 'regiao', 'C001': 'num_moradores', 
        'C010': 'conj_comp', 'C011': 'estado_civil', 'E001': 'trabalho', 'E01602': 'vl_trab_mes',
        'E01604': 'vl_est_mercad_mes', 'E01802': 'vl_trab_outros_mes', 'E01804': 'vl_trab_est_mercad_mes',
        'F00102': 'vl_aposent', 'F00702': 'vl_pens_al', 'F00802': 'vl_aluguel', 'I001': 'plano_saude',
        'J002': 'ativ_hab_saude', 'Z004': 'peso', 'Z005': 'altura', 'J037': 'internacao', 'N001': 'pcp_saude',
        'N004': 'angina_rapd', 'N005': 'angina_norm', 'N010': 'prob_sono', 'N011': 'prob_cansaco',
        'W00303': 'circ_cintura', 'Z025': 'creatinina', 'Z026': 'egfr_afro', 'Z027': 'egfr_nao_afro',
        'Z031': 'colesterol', 'Z032': 'hdl', 'Z033': 'ldl', 'Z036': 'hba1c', 'Z035': 'glicose',
        'Z044': 'rel_potassio_crea', 'Z045': 'rel_sodio_crea', 'Z049': 'calc_ingest_sal', 'P002': 'tmp_pesou',
        'P006': 'feijao_dias', 'P007': 'salada_dias', 'P009': 'verd_legu_dias', 'P011': 'carne_dias', 
        'P012': 'tipo_carne', 'P013': 'frango_dias', 'P014': 'tipo_frango', 'P015': 'peixe_dias', 
        'P016': 'suco_dias', 'P018': 'frutas_dias', 'P020': 'refri_dias', 'P021': 'tipo_refri', 
        'P023': 'leite_dias', 'P024': 'tipo_leite', 'P025': 'doces_dias', 'P026': 'subst_ref_dias', 
        'P02601': 'consumo_sal', 'P027': 'alcool', 'P034': 'exercicio', 'P035': 'exercicio_dias', 'P036': 'tipo_exercicio',
        'P038': 'anda_pe', 'P039': 'trab_esforco', 'P03901': 'trab_esf_dias', 'P03902': 'trab_esf_horas', 
        'P03903': 'trab_esf_minutos', 'P040': 'trab_pe_bk', 'P04101': 'trab_pe_bk_horas', 'P04102': 'trab_pe_bk_minutos',
        'P042': 'atv_pe_bk', 'P04301': 'atv_pe_bk_horas', 'P04302': 'atv_pe_bk_minutos', 'P044': 'atv_dom_esf',
        'P04401': 'atv_dom_esf_dias', 'P04403': 'atv_dom_esf_horas', 'P04404': 'atv_dom_esf_minutos', 
        'P045': 'tv_horas', 'P046': 'pub_exercicio', 'P050': 'fuma_atual', 'P051': 'fumo_rotina_hist', 
        'P052': 'fumo_hist', 'P053': 'idade_inic_fumar', 'P068': 'fumo_passivo', 'Q030': 'diag_diab', 
        'Q060': 'diag_colest', 'Q063': 'diag_dcore', 'Q068': 'diag_avc', 'Q132': 'med_dormir'
    }
    
    print(f'Carregar os dados do arquivo Excel...')
    dados = pd.read_excel(path)
    print(f'\nConjunto de Dados Original:')
    print(f'Tamanho do conjunto original: {dados.shape}')

    print(f'\nAjustar e filtrar variáveis de interesse...')
    dados.columns = dados.columns.str.upper()
    dados = dados[variaveis]
    
    print(f'\nRemovendo exemplos que não compõem a população de interesse...')
    
    # Filtrar os registros onde 'questionario' é igual a 1
    df1 = dados[dados['Z051'] == 1]
    print(f'Removidos {len(dados) - len(df1)} pessoas que não consentiram ou não enviaram o questionário da pesquisa.')

    # Remover registros onde 'gravida' é igual a 1 ou 3
    df2 = df1[~df1['P005'].isin([1, 3])]
    print(f'Removidas {len(df1) - len(df2)} mulheres grávidas.')
    
    # Filtrar os registros onde 'diag_ha' diferente de 2
    df3 = df2[df2['Q002'] != 2]
    print(f'Removidas {len(df2) - len(df3)} pessoas com diagnóstico de hipertensão apenas durante a gravidez.')
    
    # Filtrar os registros onde 'remed_ha' diferente de 1
    # Remover quem toma remédio para pressão
    df4 = df3[df3['Q006'] != 1]
    #print(f'Removidos {len(df3) - len(df4)} pessoas que tomam remédio para hipertensão.')
    #df4=df3.copy()
    
    # Converter colunas para tipo numérico
    df4['W00407'] = pd.to_numeric(df4['W00407'], errors='coerce')
    df4['W00408'] = pd.to_numeric(df4['W00408'], errors='coerce')
    df4['Q002'] = pd.to_numeric(df4['Q002'], errors='coerce')
    df4['Q006'] = pd.to_numeric(df4['Q006'], errors='coerce')

    print(f'\nConstrução da variável alvo: pessoas com Pressão arterial sistólica superior a 140mmHg ou Pressão arterial diastólica 90mmHg')
    
    # Criar a variável 'target' binária
    df4['target'] = ((df4['W00407'] >= 140) | (df4['W00408'] >= 90)).astype(int)
    # Ajuste da condição para incluir as novas regras
    #df4['target'] = ((df4['W00407'] >= 140) | 
    #                (df4['W00408'] >= 90) | 
    #                (df4['Q002'] == 1) | 
    #                (df4['Q006'] == 1)).astype(int)

    
    print(f'\nAjustando os dados...')
    print(f'\nRenomeando colunas')
    # Renomear as colunas
    df4.rename(columns=colunas_renomeadas, inplace=True)

    print(f'\nRemover colunas usadas no filtro da população.')
    # Remover as colunas 'questionario', 'gravida', 'remed_ha'
    df4 = df4.drop(columns=['questionario', 'gravida', 'remed_ha'])

    print(f'\nAjustar declaração de valor nulo NaN e #NULL! por np.nan.')
    # Ajustes dos valores NaN e '#NULL!' por np.nan
    df4.replace({'NaN': np.nan, '#NULL!': np.nan}, inplace=True)
    
    # Remove linhas com a região nula
    df5 = df4.dropna(subset=['regiao'])
    print(f'\nRemovendo {len(df4) - len(df5)} linhas com a região nula.')
    
    # Remove linhas com a maioria dos registros nulos
    df6 = df5.dropna(subset=['feijao_dias'])
    print(f'\nRemovendo {len(df5) - len(df6)} linhas com a maioria dos registros nulos.')
    
    # Resetar o índice do DataFrame
    df6.reset_index(drop=True, inplace=True)

    print(f'\nDivisão dos dados em Treino e Teste ...')
    # Separar variáveis independentes (X) e dependente (y)
    X = df6.drop(columns=['target'])
    y = df6['target']
    
    # Dividir os dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=teste, stratify=y, random_state=42)

    print(f'\n Conjunto de Dados de Desenvolvimento:')
    print(f'Tamanho do conjunto desenvolvimento: {df6.shape}')
    print(f'Distribuição da variável alvo no conjunto desenvolvimento:\n{df6["target"].value_counts(normalize=True) * 100}')

    print(f'\n Conjunto de Dados de Treino:')
    print(f'Tamanho do conjunto de treino: {len(X_train)}')
    print(f'Distribuição da variável alvo no treino:\n{y_train.value_counts(normalize=True) * 100}')
    # Reunir o dataset de treino em um único DataFrame
    df_train = pd.concat([X_train, y_train], axis=1)
    print(f'Formato do conjunto de dados de treino: {df_train.shape}')
    
    print(f'\n Conjunto de Dados de Teste:')
    print(f'Tamanho do conjunto de teste: {len(X_test)}')
    print(f'Distribuição da variável alvo no teste:\n{y_test.value_counts(normalize=True) * 100}')
    # Reunir o dataset de teste em um único DataFrame
    df_test = pd.concat([X_test, y_test], axis=1)
    print(f'Formato do conjunto de dados de teste: {df_test.shape}')
    
    return df6, df_train, df_test

# Função criar variáveis

def consumo_saudavel(frequencia):
    """
    Classifica a frequência de consumo de um alimento durante a semana.

    Parâmetros:
    frequencia (int): Frequência de consumo do alimento durante a semana.
    - 1: Muito Saudável: 7 vezes por semana
    - 2: Saudável: 5 a 6 vezes por semana
    - 3: Moderadamente Saudável: 3 a 4 vezes por semana
    - 4: Pouco Saudável: 0 a 2 vezes por semana

    Retorna:
    str: Categoria de consumo.
    """
    if frequencia == 7:
        return '1'
    elif 5 <= frequencia <= 6:
        return '2'
    elif 3 <= frequencia <= 4:
        return '3'
    else:
        return '4'

def consumo_carne_frango(frequencia, tipo):
    """
    Classifica a frequência de consumo de carne durante a semana, considerando seu tipo.

    Parâmetros:
    - 1: Muito Saudável: 1 a 2 vezes por semana, sem gordura
    - 2: Saudável: 3 a 4 vezes por semana, sem gordura
    - 3: Moderadamente Saudável: 1 a 4 vezes por semana, com gordura
    - 4: Não Saudável: 5 a 7 vezes por semana, com gordura

    Retorna:
    str: Categoria de consumo.
    """
    
    # Tratando casos onde tipo é nulo e a frequência é 0
    if pd.isna(tipo) and frequencia == 0:
        tipo = 1

    # Cobrir todas as combinações possíveis
    if 0 <= frequencia <= 2:
        if tipo == 1 or pd.isna(tipo):
            return '1'
        elif tipo == 2:
            return '3'
    elif 3 <= frequencia <= 4:
        if tipo == 1:
            return '2'
        elif tipo == 2:
            return '3'
    elif 5 <= frequencia <= 7:
        if tipo == 1:
            return '3'
        elif tipo == 2:
            return '4'
    else:
        return '4'

def consumo_peixe(row):
    """
    Classifica a frequência de consumo de peixe durante a semana.

    Parâmetros:
    row (Series): Linha do DataFrame contendo a coluna 'peixe'.

    Retorna:
    str: Categoria de consumo.
    """
    frequencia = row['peixe_dias']

    if frequencia >= 3:
        return '1'
    elif frequencia == 2:
        return '2'
    elif frequencia == 1:
        return '3'
    elif frequencia == 0:
        return '4'

def consumo_suco_frutas(frequencia):
    """
    Classifica a frequência de consumo de um alimento durante a semana.

    Parâmetros:
    frequencia (int): Frequência de consumo do alimento durante a semana.

    Retorna:
    str: Categoria de consumo.
    """
    if 0 <= frequencia <= 1:
        return '1'
    elif 2 <= frequencia <= 3:
        return '2'
    elif 4 <= frequencia <= 5:
        return '3'
    elif 6 <= frequencia <= 7:
        return '4'
    else:
        return 'Frequência Inválida'   
    
def consumo_doce(frequencia):
    """
    Classifica a frequência de consumo de um alimento durante a semana.

    Parâmetros:
    frequencia (int): Frequência de consumo do alimento durante a semana.

    Retorna:
    str: Categoria de consumo.
    """
    if frequencia == 0:
        return '1'
    elif frequencia == 1:
        return '2'
    elif 2 <= frequencia <= 3:
        return '3'
    elif frequencia >= 4:
        return '4'
    else:
        return 'Frequência Inválida' 


# Classificar nível de atividade física
def calcular_tempo_total_atividade_fisica(row):
    total_minutos = 0
    
    # Tempo de exercício físico
    if row['exercicio'] == 1:
        total_minutos += row['exercicio_dias'] * 30  # Assumindo 30 minutos por dia de exercício
    
    # Tempo de caminhada a pé
    if row['atv_pe_bk'] > 0:
        total_minutos += row['atv_pe_bk'] * (
            row['atv_pe_bk_horas'] * 60 + row['atv_pe_bk_minutos']
        )
    
    # Tempo de trabalho com esforço físico
    if row['trab_esforco'] == 1:
        total_minutos += row['trab_esf_dias'] * (
            row['trab_esf_horas'] * 60 + row['trab_esf_minutos']
        )
    
    # Tempo de deslocamento a pé ou de bicicleta
    if row['trab_pe_bk'] in [1, 2]:
        total_minutos += row['trab_pe_bk_horas'] * 60 + row['trab_pe_bk_minutos']
    
    # Tempo de atividades domésticas
    if row['atv_dom_esf'] == 1:
        total_minutos += row['atv_dom_esf_dias'] * (
            row['atv_dom_esf_horas'] * 60 + row['atv_dom_esf_minutos']
        )
    
    return total_minutos

# Função para classificar o nível de atividade física
def classificar_nivel_atividade_fisica(row):
    total_minutos = row['total_minutos_atividade_fisica']
    tipo_exercicio = row['tipo_exercicio']
    
    # Tipos de exercício compatíveis com atividade moderada e vigorosa
    tipos_moderada = {1, 2, 7, 8, 11, 16}
    tipos_vigorosa = {3, 4, 5, 6, 9, 10, 12, 13, 14, 15}
    
    if total_minutos == 0:
        return '1'
    elif total_minutos < 150:
        return '2'
    elif row['exercicio'] == 1:
        if tipo_exercicio in tipos_vigorosa and total_minutos >= 75:
            return '4'
        elif tipo_exercicio in tipos_moderada and total_minutos >= 150:
            return '3'
        elif tipo_exercicio in tipos_vigorosa and total_minutos >= 150:
            return '3'
        elif tipo_exercicio in tipos_moderada and total_minutos >= 300:
            return '4'
    elif row['atv_dom_esf'] == 1 and total_minutos >= 150:
        return '3'
    return '2'


def impute_knn(df, numeric_var, categorical_vars):
    # Separar a variável numérica e as variáveis categóricas
    numeric_data = df[[numeric_var]]
    categorical_data = df[categorical_vars]
    
    # Aplicar OneHotEncoder nas variáveis categóricas
    encoder = OneHotEncoder(drop='first', sparse=False)
    categorical_data_encoded = encoder.fit_transform(categorical_data)
    
    # Concatenar a variável numérica e as variáveis categóricas codificadas
    imputer_data = np.concatenate([numeric_data, categorical_data_encoded], axis=1)
    
    # Inicializar o KNNImputer
    imputer = KNNImputer(n_neighbors=5)
    
    # Aplicar o KNNImputer
    imputed_data = imputer.fit_transform(imputer_data)
    
    # Extrair a variável numérica imputada
    numeric_imputed = imputed_data[:, 0]
    
    # Adicionar a variável imputada ao dataframe original
    #df[numeric_var] = numeric_imputed
    
    return numeric_imputed

# Criando a variável de classificação do IMC
def classificar_imc(imc):
    if imc >= 40.00:
        return '4'
    elif imc >= 30.00:
        return '3'
    elif imc >= 25.00:
        return '2'
    elif imc >= 18.5:
        return '1'
    else:
        return '5'

def classificar_substituir_ref(frequencia):
    """
    Classifica a frequência de consumo de um alimento durante a semana.

    Parâmetros:
    frequencia (int): Frequência de consumo do alimento durante a semana.

    Retorna:
    str: Categoria de consumo.
    """
    if frequencia == 0:
        return '1'
    elif frequencia == 1:
        return '2'
    elif frequencia == 2:
        return '3'
    elif frequencia >= 3:
        return '4'
    else:
        return 'Frequência Inválida' 


# Função que executa as criações
def add_variaveis_ajustadas(df):

    print(f'\nVariáveis Demográficas ...')
    print(f'\n    Faixa Etária:')
    bins = [18, 30, 40, 50, 60, 70, 80, 105]
    labels = ['18-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80+']
    df['faixa_etaria'] = pd.cut(df['idade'], bins=bins, labels=labels, right=False)
    print(f'Distribuição da Faixa Etária:\n{df["faixa_etaria"].value_counts(normalize=True) * 100}')

    print(f'\n    Número de Moradores:')
    bins_moradores = [1, 2, 3, 4, 5, 6, float('inf')]
    labels_moradores = ['1', '2', '3', '4', '5', '6+']
    df['faixa_num_mor'] = pd.cut(df['num_moradores'], bins=bins_moradores, labels=labels_moradores, right=False)
    print(f'Distribuição do Número de Moradores:\n{df["faixa_num_mor"].value_counts(normalize=True) * 100}')
    
    print(f'\n    Estado Civil:')
    df['estado_civil_ag'] = df['estado_civil'].replace({
        1.0: '1',
        2.0: '3',
        3.0: '3',
        4.0: '4',
        5.0: '2'
    })
    print(f'Distribuição do Estado Civil:\n{df["estado_civil_ag"].value_counts(normalize=True) * 100}')

    print(f'\n    Etnia negra:')
    df['etnia_negra'] = df['etnia'].replace({
        1.0: '2',
        2.0: '1',
        3.0: '2',
        4.0: '1',
        5.0: '2',
        9.0: '2'
    })
    print(f'Distribuição do Etnia negra:\n{df["etnia_negra"].value_counts(normalize=True) * 100}')
    
    print(f'\nVariáveis Socioeconômicas ...')
    print(f'\n    Renda Total:')
    df['renda_total'] = df[
        ['vl_trab_mes', 'vl_est_mercad_mes', 'vl_trab_outros_mes', 'vl_trab_est_mercad_mes', 'vl_aposent', 'vl_pens_al', 'vl_aluguel']
    ].sum(axis=1, skipna=True)
    print(f'Distribuição do Renda Total:\n{df["renda_total"].describe()}')
    salario_minimo = 678.00
    bins = [
        0, 
        0.25 * salario_minimo, 
        0.5 * salario_minimo, 
        1 * salario_minimo, 
        2 * salario_minimo, 
        3 * salario_minimo, 
        5 * salario_minimo, 
        10 * salario_minimo, 
        20 * salario_minimo, 
        float('inf')
    ]
    labels = ['1', '2', '3', '4', '5','6','7','8','9']
    df['faixa_renda_sl'] = pd.cut(df['renda_total'], bins=bins, labels=labels, right=False)
    print(f'Distribuição da Faixa de renda por Salário Mínimo:\n{df["faixa_renda_sl"].value_counts(normalize=True) * 100}')

    print(f'\nVariáveis Doenças Preexistentes ...')
    print(f'\n    Diagnóstico de Diabetes:')
    # diagnóstico de diabetes
    df['diag_diab_rec'] = df['diag_diab'].replace(3, 2)
    df['diag_diab_rec'] = df['diag_diab_rec'].fillna(3.0)
    print(f'Distribuição do Diagnóstico de Diabetes:\n{df["diag_diab_rec"].value_counts(normalize=True) * 100}')

    print(f'\n    Diagnóstico de Colesterol Alto:')
    df['diag_colest_rec'] = df['diag_colest'].fillna(3.0)
    print(f'Distribuição do Diagnóstico de Colesterol Alto:\n{df["diag_colest_rec"].value_counts(normalize=True) * 100}')

    print(f'\n    Diagnóstico de uma Doença do Coração:')
    df['diag_dcore'] = df['diag_dcore'].fillna(2.0)
    print(f'Distribuição do Diagnóstico de uma Doença do Coração:\n{df["diag_dcore"].value_counts(normalize=True) * 100}')

    print(f'\n    Diagnóstico de uma Diagnóstico de AVC:')
    df['diag_avc'] = df['diag_avc'].fillna(2.0)
    print(f'Distribuição do Diagnóstico de uma Diagnóstico de AVC:\n{df["diag_avc"].value_counts(normalize=True) * 100}')
    
    print(f'\nVariáveis Estilos de Vida ...')
    
    print(f'\n    Consumo de Feijão:')
    df['feijao_consumo'] = df['feijao_dias'].apply(consumo_saudavel)
    print(f'Distribuição do Consumo de Feijão:\n{df["feijao_consumo"].value_counts(normalize=True) * 100}')

    print(f'\n    Consumo de Salada:')
    df['salada_consumo'] = df['salada_dias'].apply(consumo_saudavel)
    print(f'Distribuição do Consumo de Salada:\n{df["salada_consumo"].value_counts(normalize=True) * 100}')

    print(f'\n    Consumo de Verduras e Legumes:')
    df['verd_legu_consumo'] = df['verd_legu_dias'].apply(consumo_saudavel)
    print(f'Distribuição do Consumo de Verduras e Legumes:\n{df["verd_legu_consumo"].value_counts(normalize=True) * 100}')

    print(f'\n    Consumo Carne Vermelha:')
    df['carne_consumo'] = df.apply(lambda row: consumo_carne_frango(row['carne_dias'], row['tipo_carne']), axis=1)
    print(f'Distribuição do Consumo Carne Vermelha:\n{df["carne_consumo"].value_counts(normalize=True) * 100}')

    print(f'\n    Consumo Frango:')
    df['frango_consumo'] = df.apply(lambda row: consumo_carne_frango(row['frango_dias'], row['tipo_frango']), axis=1)
    print(f'Distribuição do Consumo Frango:\n{df["frango_consumo"].value_counts(normalize=True) * 100}')

    print(f'\n    Consumo Peixe:')
    df['peixe_consumo'] = df.apply(consumo_peixe, axis=1)
    print(f'Distribuição do Consumo Peixe:\n{df["peixe_consumo"].value_counts(normalize=True) * 100}')

    print(f'\n    Consumo Suco de frutas natural:')
    df['suco_consumo'] = df['suco_dias'].apply(consumo_suco_frutas)
    print(f'Distribuição do Consumo Suco de frutas natural:\n{df["suco_consumo"].value_counts(normalize=True) * 100}')

    print(f'\n    Consumo Fruta:')
    df['frutas_consumo'] = df['frutas_dias'].apply(consumo_saudavel)
    print(f'Distribuição do Consumo Suco de frutas natural:\n{df["frutas_consumo"].value_counts(normalize=True) * 100}')

    print(f'\n    Consumo Refrigerante ou suco artificial:')
    df['tipo_refri_rec'] = df['tipo_refri'].fillna(4.0)
    print(f'Distribuição do Consumo Suco de frutas natural:\n{df["tipo_refri_rec"].value_counts(normalize=True) * 100}')

    print(f'\n    Consumo Leite:')
    df['tipo_leite_rec'] = df['tipo_leite'].fillna(4.0)
    print(f'Distribuição do Consumo Leite:\n{df["tipo_leite_rec"].value_counts(normalize=True) * 100}')

    print(f'\n    Consumo Doce:')
    df['doces_consumo'] = df['doces_dias'].apply(consumo_doce)
    print(f'Distribuição do Consumo Doce:\n{df["doces_consumo"].value_counts(normalize=True) * 100}')

    print(f'\n    Substituição a refeição do almoço ou jantar por lanches:')
    df['freq_subst_ref'] = df['subst_ref_dias'].apply(classificar_substituir_ref)
    print(f'Distribuição do Substituição a refeição:\n{df["freq_subst_ref"].value_counts(normalize=True) * 100}')

    # Consumo de sal
    print(f'\n    Consumo de sal:')
    df['consumo_sal_rec'] = df['consumo_sal'].replace({
        1: '3',
        2: '3',
        3: '2',
        4: '1',
        5: '1',
    })
    print(f'Distribuição do Consumo Sal:\n{df["consumo_sal_rec"].value_counts(normalize=True) * 100}')

    print(f'\n    Nível de atividade física:')
    df['total_minutos_atividade_fisica'] = df.apply(calcular_tempo_total_atividade_fisica, axis=1)
    print(f'Distribuição do Total minutos em atividade física :\n{df["total_minutos_atividade_fisica"].describe()}')
    
    df['nivel_atv_fisica'] = df.apply(classificar_nivel_atividade_fisica, axis=1)
    print(f'Distribuição do Nível atividade física:\n{df["nivel_atv_fisica"].value_counts(normalize=True) * 100}')

    print(f'\n    Fumante:')
    df['fumante'] = df['fuma_atual'].replace({
        1: '1',
        2: '1',
        3: '2',
        '.': '2'
    })
    print(f'Distribuição do Fumante:\n{df["fumante"].value_counts(normalize=True) * 100}')

    print(f'\n    Histórico de tabagismo:')
    df['fumante_hist'] = df.apply(
        lambda row: 1 if row['fumante'] == 1 or row['fumo_rotina_hist'] == 1 or row['fumo_hist'] == 1 else 2,
        axis=1
    )
    print(f'Distribuição do Histórico de tabagismo:\n{df["fumante_hist"].value_counts(normalize=True) * 100}')

    print(f'\n    Classificação IMC:')
    df['altura_m'] = df['altura'] / 100
    df['imc'] = df['peso'] / (df['altura_m'] ** 2)
    df['class_imc'] = df['imc'].apply(classificar_imc)
    print(f'Distribuição do Classificação IMC:\n{df["class_imc"].value_counts(normalize=True) * 100}')


    print(f'\n    Circunferência da cintura:')
    df['cintura_risco_aumentado'] = np.where(
        ((df['sexo'] == 2) & (df['circ_cintura'] > 80)) | 
        ((df['sexo'] == 1) & (df['circ_cintura'] > 94)), 
        '1', 
        '2'
    )
    df['cintura_risco_muito_aumentado'] = np.where(
        ((df['sexo'] == 2) & (df['circ_cintura'] > 88)) | 
        ((df['sexo'] == 1) & (df['circ_cintura'] > 102)), 
        '1', 
        '2'
    )
    print(f'Distribuição do Risco aumentado:\n{df["cintura_risco_aumentado"].value_counts(normalize=True) * 100}')
    print(f'Distribuição do Risco muito aumentado:\n{df["cintura_risco_muito_aumentado"].value_counts(normalize=True) * 100}')

    print(f'\nVariáveis Saúde ...')
    
    print(f'\n    Percepção da Saúde:')
    df['pcp_saude_rec'] = df['pcp_saude'].replace({
        1: '2',
        2: '2',
        3: '2',
        4: '1',
        5: '1',
    })
    print(f'Distribuição da Percepção da Saúde:\n{df["pcp_saude_rec"].value_counts(normalize=True) * 100}')


    print(f'\n    Angina:')
    df['angina'] = np.where((df['angina_rapd'] == 1) & (df['angina_norm'] == 1), 1, 2)
    print(f'Distribuição da Angina:\n{df["angina"].value_counts(normalize=True) * 100}')

    print(f'\nVariáveis Exames Laboratoriais ...')
    # Imputar valores
    categorical_vars = ['faixa_etaria', 'sexo', 'regiao', 
                        'faixa_renda_sl', 'nivel_atv_fisica', 'etnia',
                        'diag_renalc', 'diag_diab_rec', 'cintura_risco_aumentado',
                        'diag_colest_rec', 'diag_dcore', 'med_dormir']

    
    print(f'\n    Filtração glomerular afrodescendente:')
    df['egfr_afro_imp'] = impute_knn(df, 'egfr_afro', categorical_vars)
    bins = [0, 30, 45, 60, 90, np.inf]
    labels = ['5', '4', '3', '2', '1']
    df['cat_egfr_afro'] = pd.cut(df['egfr_afro_imp'], bins=bins, labels=labels, right=False)
    print(f'Distribuição da Filtração glomerular afrodescendente por categoria:\n{df["cat_egfr_afro"].value_counts(normalize=True) * 100}')

    print(f'\n    Colesterol:')
    df['colesterol_imp'] = impute_knn(df, 'colesterol', categorical_vars)
    df['colesterol_ideal'] = df['colesterol_imp'].apply(lambda x: '1' if x < 190 else '2')
    print(f'Distribuição da Colesterol por categoria:\n{df["colesterol_ideal"].value_counts(normalize=True) * 100}')

    print(f'\n    Glicose:')
    df['glicose_imp'] = impute_knn(df, 'glicose', categorical_vars)
    bins = [0, 70, 100, 126, np.inf]
    labels = ['4', '1', '2', '3']
    df['cat_glicose'] = pd.cut(df['glicose_imp'], bins=bins, labels=labels, right=False)
    print(f'Distribuição da Colesterol por categoria:\n{df["cat_glicose"].value_counts(normalize=True) * 100}')
    
    return df

# Coeficiente de Correlação de Cramér's V
def cramers_v(x, y):
    """
    Calcula o Coeficiente de Correlação de Cramér's V entre duas variáveis categóricas.
    """
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))


def calc_iv(df, feature, target, eps=1e-10):
    """
    Calcula o Information Value (IV) para uma variável categórica em relação ao target binário.
    """
    lst = []
    total_good = (df[target] == 0).sum()
    total_bad = (df[target] == 1).sum()
    
    for val in df[feature].unique():
        if pd.isna(val):
            continue
        good = df[(df[feature] == val) & (df[target] == 0)].shape[0]
        bad = df[(df[feature] == val) & (df[target] == 1)].shape[0]
        dist_good = (good + eps) / (total_good + eps)
        dist_bad = (bad + eps) / (total_bad + eps)
        woe = np.log(dist_good / dist_bad)
        iv = (dist_good - dist_bad) * woe
        lst.append(iv)
    
    return np.sum(lst)

def classify_cramers_v(value):
    """
    Classifica o Coeficiente de Correlação de Cramér's V.
    """
    if 0 <= value < 0.1:
        return 'Associação muito fraca'
    elif 0.1 <= value < 0.3:
        return 'Associação fraca'
    elif 0.3 <= value < 0.5:
        return 'Associação moderada'
    elif 0.5 <= value < 0.7:
        return 'Associação forte'
    elif 0.7 <= value <= 1:
        return 'Associação muito forte'
    else:
        return 'Fora do intervalo esperado'

def classify_iv(value):
    """
    Classifica o Information Value (IV).
    """
    if value < 0.02:
        return 'Não útil para predição'
    elif 0.02 <= value < 0.1:
        return 'Previsão fraca'
    elif 0.1 <= value < 0.3:
        return 'Previsão média'
    elif value >= 0.3:
        return 'Forte previsão'
    else:
        return 'Fora do intervalo esperado'


def summarize_categorical_features(df, variaveis, target):
    """
    Resume variáveis categóricas com proporção de valores nulos, Cramér's V e Information Value.
    
    Parâmetros:
    - df (pd.DataFrame): DataFrame contendo as variáveis categóricas e a variável target binária.
    - target (str): Nome da coluna target binária.

    Retorna:
    - pd.DataFrame: DataFrame contendo a proporção de valores nulos, Cramér's V e Information Value.
    """
    df = df[variaveis + ['target']]

    # Converter todas as colunas float para inteiro
    for col in df.select_dtypes(include=['float']):
        df[col] = df[col].astype('Int64')
    
    summary = []

    for col in df.columns:
        if col == target:
            continue
        null_ratio = df[col].isna().mean()
        cramers_v_value = cramers_v(df[col], df[target])
        iv_value = calc_iv(df, col, target)
        
        summary.append({
            'Variable': col,
            'Null Proportion': null_ratio,
            'Cramér\'s V': cramers_v_value,
            'Information Value': iv_value,
            'Cramér\'s V Classification': classify_cramers_v(cramers_v_value),
            'IV Classification': classify_iv(iv_value)
        })
    
    return pd.DataFrame(summary)

def get_significant_variables(summary_df):
    """
    Retorna uma lista de variáveis que não receberam a classificação 
    'Associação muito fraca' e 'Não útil para predição' simultaneamente.

    Parâmetros:
    - summary_df (pd.DataFrame): DataFrame de resumo contendo as classificações.

    Retorna:
    - List: Lista de variáveis significativas.
    """
    filtered_df = summary_df[
        ~((summary_df["Cramér's V Classification"] == 'Associação muito fraca') & 
          (summary_df["IV Classification"] == 'Não útil para predição'))
    ]
    return filtered_df['Variable'].tolist()

def cramers_corrected_stat(confusion_matrix):
    """Calcula o Coeficiente de Correlação de Cramér's V corrigido."""
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1)) / (n-1))
    rcorr = r - ((r-1)**2) / (n-1)
    kcorr = k - ((k-1)**2) / (n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))


def calculate_cramers_v(df, variaveis):
    """
    Calcula o Coeficiente de Correlação de Cramér's V entre variáveis categóricas.

    Parâmetros:
    - df (pd.DataFrame): DataFrame contendo as variáveis categóricas.

    Retorna:
    - tuple: Duas listas, uma com as variáveis sem pares acima de 0.5 e outra com os pares de variáveis com coeficiente acima de 0.5.
    """
    df = df[variaveis]
    
    # Converter todas as colunas para o tipo categórico
    df = df.apply(lambda x: x.astype('category') if x.dtype == 'O' else x)

    # Lista para armazenar os resultados
    high_corr_pairs = []
    low_corr_vars = set(df.columns)

    # Calcular Cramér's V para cada par de variáveis
    for col1 in df.columns:
        for col2 in df.columns:
            if col1 != col2:
                confusion_matrix = pd.crosstab(df[col1], df[col2])
                cramers_v = cramers_corrected_stat(confusion_matrix)
                if cramers_v > 0.5:
                    high_corr_pairs.append((col1, col2, cramers_v))
                    if col1 in low_corr_vars:
                        low_corr_vars.remove(col1)
                    if col2 in low_corr_vars:
                        low_corr_vars.remove(col2)

    return list(low_corr_vars), high_corr_pairs


# Balancear os dados via SMOTE

def balance_data_with_smote(df, target):
    """
    Aplica SMOTE para balancear os dados.

    Parâmetros:
    - df (pd.DataFrame): DataFrame contendo as variáveis significativas e a variável target binária.
    - target (str): Nome da coluna target binária.

    Retorna:
    - pd.DataFrame: DataFrame balanceado.
    """
    # Copiar o DataFrame para evitar alterações no original
    X = df.drop(target, axis=1).astype('category')
    y = df[target]
    
    # Aplicar OneHotEncoder nas colunas categóricas
    encoder = OneHotEncoder(drop='first', sparse=False)
    X_encoded = encoder.fit_transform(X)

    # Criar um DataFrame com as variáveis codificadas
    X_encoded_df = pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out())
        
    # Aplicar SMote para balancear os dados
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_encoded_df, y)

    # Arredondar todas as variáveis para garantir que sejam binárias
    X_resampled = X_resampled.round().astype(int)
    
    # Combinar X_resampled e y_resampled em um DataFrame
    df_resampled = pd.concat([X_resampled, y_resampled], axis=1)
    
    return df_resampled

def encode_data(df, target):
    """
    Aplica OneHotEncoder nas variáveis categóricas e retorna o DataFrame resultante com valores inteiros.

    Parâmetros:
    - df (pd.DataFrame): DataFrame contendo as variáveis significativas e a variável target binária.
    - target (str): Nome da coluna target binária.

    Retorna:
    - pd.DataFrame: DataFrame com as variáveis codificadas.
    """
    # Copiar o DataFrame para evitar alterações no original
    X = df.drop(target, axis=1).astype('category')
    y = df[target]
    
    # Aplicar OneHotEncoder nas colunas categóricas
    encoder = OneHotEncoder(drop='first', sparse=False)
    X_encoded = encoder.fit_transform(X)

    # Criar um DataFrame com as variáveis codificadas
    X_encoded_df = pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out())
    
    # Converter os valores para inteiros
    X_encoded_df = X_encoded_df.astype(int)
    
    # Combinar X_encoded_df e y em um DataFrame
    df_encoded = pd.concat([X_encoded_df, y.reset_index(drop=True)], axis=1)
    
    return df_encoded

def variable_selection_cv(df, target):

    X = df.drop(columns=[target])
    y = df[target]

    # Definir parâmetros dos modelos
    xgb_params = {'random_state': 42}
    xgb = XGBClassifier(**xgb_params)

    # Lista de etapas
    steps = [
        "Boruta"
    ]

    # Inicializar a barra de progresso
    with tqdm(total=len(steps), desc="Seleção de Variáveis", unit="etapa") as pbar:
        # Boruta com RandomForest
        np.int = np.int32
        np.float = np.float64
        np.bool = np.bool_
        
        # Aplicar SMOTE apenas para Boruta
        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(X, y)
        # Arredondar todas as variáveis para garantir que sejam binárias
        X_res = X_res.round().astype(int)
        
        pbar.set_description("Executando Boruta XGB")
        boruta_selector_xgb = BorutaPy(xgb, n_estimators='auto', random_state=42)
        boruta_selector_xgb.fit(X_res.values, y_res.values)
        boruta_support_xgb = boruta_selector_xgb.support_
        pbar.update(1)


    # Criar DataFrame com as variáveis selecionadas
    selected_features = pd.DataFrame({
        'Variable': X.columns,
        'Boruta': boruta_support_xgb,
    })

    return selected_features

# Função para avaliar os modelos
def evaluate_models(train_df, test_df, target):
    X_train = train_df.drop(columns=[target])
    y_train = train_df[target]
    X_test = test_df.drop(columns=[target])
    y_test = test_df[target]

    models = {
        'LogisticRegression': LogisticRegression(C=1.0, solver='liblinear', class_weight='balanced', random_state=42),
        'NaiveBayes': GaussianNB(),
        'KNN': KNeighborsClassifier(n_neighbors=5, weights='uniform', p=2),
        'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1, class_weight='balanced', random_state=42),
        'XGBoost': XGBClassifier(learning_rate=0.1, max_depth=6, min_child_weight=1, gamma=0, colsample_bytree=0.8, n_estimators=100, scale_pos_weight=np.sum(y_train == 0) / np.sum(y_train == 1), random_state=42),
        'LightGBM': LGBMClassifier(learning_rate=0.1, max_depth=-1, num_leaves=31, min_child_samples=20, subsample=1.0, colsample_bytree=1.0, n_estimators=100, class_weight='balanced', verbosity=-1, random_state=42),
        'CatBoost': CatBoostClassifier(learning_rate=0.1, depth=6, l2_leaf_reg=3, bagging_temperature=1, random_strength=1, n_estimators=100, class_weights=[1, np.sum(y_train == 0) / np.sum(y_train == 1)], random_state=42, silent=True),
        'SVM': SVC(C=1.0, kernel='rbf', gamma='scale', class_weight='balanced', probability=True, random_state=42),
        'NeuralNetwork': MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', alpha=0.0001, learning_rate='constant', random_state=42, max_iter=1000)
    }

    results = []
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    plt.figure(figsize=(16, 10))
    ax = plt.gca()

    for model_name, model in tqdm(models.items(), desc="Avaliando Modelos"):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = {
            'Model': model_name,
            'AUC': round(roc_auc_score(y_test, y_prob), 2),
            'Accuracy': round(accuracy_score(y_test, y_pred), 2),
            'Precision': round(precision_score(y_test, y_pred), 2),
            'Recall': round(recall_score(y_test, y_pred), 2),
            'F1': round(f1_score(y_test, y_pred), 2)
        }
        
        results.append(metrics)

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        ax.plot(fpr, tpr, lw=2, alpha=0.8, label=f'{model_name} (AUC = {roc_auc:.2f})')

    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='gray', alpha=.8)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Taxa de Falsos Positivos')
    ax.set_ylabel('Taxa de Verdadeiros Positivos')
    ax.set_title('Receiver Operating Characteristic')
    ax.legend(loc="lower right")
    plt.show()

    results_df = pd.DataFrame(results)
    return results_df


def evaluate_models_cv(df, target, cv=5):
    X = df.drop(columns=[target])
    y = df[target]

    models = {
        'LogisticRegression': LogisticRegression(C=1.0, solver='liblinear', class_weight='balanced', random_state=42),
        'NaiveBayes': GaussianNB(),
        'KNN': KNeighborsClassifier(n_neighbors=5, weights='uniform', p=2),
        'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1, class_weight='balanced', random_state=42),
        'XGBoost': XGBClassifier(learning_rate=0.1, max_depth=6, min_child_weight=1, gamma=0, colsample_bytree=0.8, n_estimators=100, scale_pos_weight=np.sum(y == 0) / np.sum(y == 1), random_state=42),
        'LightGBM': LGBMClassifier(learning_rate=0.1, max_depth=-1, num_leaves=31, min_child_samples=20, subsample=1.0, colsample_bytree=1.0, n_estimators=100, class_weight='balanced', verbosity=-1, random_state=42),
        
        'CatBoost': CatBoostClassifier(learning_rate=0.1, depth=6, l2_leaf_reg=3, bagging_temperature=1, random_strength=1, n_estimators=100, class_weights=[1, np.sum(y == 0) / np.sum(y == 1)], random_state=42, silent=True),
        
        'SVM': SVC(C=1.0, kernel='rbf', gamma='scale', class_weight='balanced', probability=True, random_state=42),
        'NeuralNetwork': MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', alpha=0.0001, learning_rate='constant', random_state=42, max_iter=1000)
    }

    results = []
    mean_fpr = np.linspace(0, 1, 100)

    plt.figure(figsize=(16, 10))
    ax = plt.gca()

    for model_name, model in tqdm(models.items(), desc="Avaliando Modelos"):
        tprs = []
        aucs = []
        accuracies = []
        precisions = []
        recalls = []
        f1s = []
        
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            fpr, tpr, _ = roc_curve(y_test, y_prob)
            tprs.append(np.interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)

            accuracies.append(accuracy_score(y_test, y_pred))
            precisions.append(precision_score(y_test, y_pred))
            recalls.append(recall_score(y_test, y_pred))
            f1s.append(f1_score(y_test, y_pred))
        
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)
        ax.plot(mean_fpr, mean_tpr, lw=2, alpha=0.8, label=f'{model_name} (AUC = {mean_auc:.2f} ± {std_auc:.2f})')

        metrics = {
            'Model': model_name,
            'AUC': round(mean_auc, 2),
            'AUC_STD': round(std_auc, 2),
            'Accuracy': round(np.mean(accuracies), 2),
            'Accuracy_STD': round(np.std(accuracies), 2),
            'Precision': round(np.mean(precisions), 2),
            'Precision_STD': round(np.std(precisions), 2),
            'Recall': round(np.mean(recalls), 2),
            'Recall_STD': round(np.std(recalls), 2),
            'F1': round(np.mean(f1s), 2),
            'F1_STD': round(np.std(f1s), 2)
        }
        
        results.append(metrics)

    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='gray', alpha=.8)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Taxa de Falsos Positivos')
    ax.set_ylabel('Taxa de Verdadeiros Positivos')
    ax.set_title('Receiver Operating Characteristic')
    ax.legend(loc="lower right")
    plt.show()

    results_df = pd.DataFrame(results)
    return results_df


def train_catboost_model(X_train, y_train):
    # Treinar CatBoost
    catboost_model = CatBoostClassifier(
        learning_rate=0.1, depth=6, l2_leaf_reg=3, bagging_temperature=1, random_strength=1,
        n_estimators=100, class_weights=[1, np.sum(y_train == 0) / np.sum(y_train == 1)],
        random_state=42, silent=True
    )
    catboost_model.fit(X_train, y_train, verbose=False)
    return catboost_model

def plot_shap_values_catboost(catboost_model, X_test):
    # Calcular valores SHAP para CatBoost
    explainer_catboost = shap.TreeExplainer(catboost_model)
    shap_values_catboost = explainer_catboost.shap_values(X_test)

    # Plotar valores SHAP para CatBoost
    shap.summary_plot(shap_values_catboost, X_test, feature_names=X_test.columns)




# Ajustar dados de teste

def prepare_test_dataframe(df, target, variaveis):
    """
    Prepara o DataFrame de teste para avaliação de um modelo.

    Parâmetros:
    - df (pd.DataFrame): DataFrame contendo os dados de teste.
    - target (str): Nome da coluna target binária.
    - selected_features (list): Lista de variáveis a serem selecionadas.

    Retorna:
    - pd.DataFrame: DataFrame preparado com as variáveis selecionadas e codificadas.
    """

    df1 = df[variaveis + [target]]

    # Converter todas as colunas float para inteiro
    for col in df1.select_dtypes(include=['float']):
        df1[col] = df1[col].astype('Int64')

    # Copiar o DataFrame para evitar alterações no original
    X = df1.drop(target, axis=1).astype('category')
    y = df1[target]

    # Aplicar OneHotEncoder nas colunas categóricas
    encoder = OneHotEncoder(drop='first', sparse=False)
    X_encoded = encoder.fit_transform(X)

    # Criar um DataFrame com as variáveis codificadas
    X_encoded_df = pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out(), index=df.index)

    # Arredondar todas as variáveis para garantir que sejam binárias
    X_encoded_df2 = X_encoded_df.round().astype(int)
    
    # Concatenar o DataFrame codificado com a coluna target
    df2 = pd.concat([X_encoded_df2, y], axis=1)

    # Renomear as colunas
    df2.rename(columns={
        'regiao_2': 'regiao_2.0',
        'regiao_3': 'regiao_3.0',
        'regiao_4': 'regiao_4.0',
        'regiao_5': 'regiao_5.0',
        'med_dormir_2': 'med_dormir_2.0',
        'tipo_refri_rec_3': 'tipo_refri_rec_3.0', 
        'diag_dcore_2': 'diag_dcore_2.0', 
        'tipo_leite_rec_2': 'tipo_leite_rec_2.0', 
        'alcool_2': 'alcool_2.0', 
        'diag_diab_rec_2': 'diag_diab_rec_2.0', 
        'diag_diab_rec_3': 'diag_diab_rec_3.0',
        'tipo_refri_rec_2': 'tipo_refri_rec_2.0', 
        'tipo_refri_rec_4': 'tipo_refri_rec_4.0', 
        'tipo_leite_rec_3': 'tipo_leite_rec_3.0', 
        'tipo_leite_rec_4': 'tipo_leite_rec_4.0', 
        'alcool_3': 'alcool_3.0',
        'diag_renalc_2': 'diag_renalc_2.0'
    }, inplace=True)

    return df2


def train_logistic_model(X_train, y_train):
    # Escalar os dados para Regressão Logística
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Treinar Regressão Logística
    logistic_model = LogisticRegression(C=1.0, solver='liblinear', class_weight='balanced', random_state=42)
    logistic_model.fit(X_train_scaled, y_train)

    return logistic_model, scaler


def interpret_logistic_model(logistic_model, X_train):
    # Obter os coeficientes do modelo
    coef = logistic_model.coef_[0]
    feature_names = X_train.columns

    # Criar um DataFrame para facilitar a interpretação
    coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coef})
    coef_df['Exp(Coefficient)'] = np.exp(coef_df['Coefficient'])
    
    # Selecionar as 20 variáveis mais relevantes (maiores coeficientes em valor absoluto)
    coef_df['Abs(Coefficient)'] = coef_df['Coefficient'].abs()
    top_coef_df = coef_df.sort_values(by='Abs(Coefficient)', ascending=False).head(20)
    top_coef_df = top_coef_df.sort_values(by='Coefficient', ascending=True)
    
    # Plotar os coeficientes
    plt.figure(figsize=(10, 8))
    plt.barh(top_coef_df['Feature'], top_coef_df['Coefficient'], color='blue')
    plt.xlabel('Coefficient')
    plt.ylabel('Feature')
    plt.title('Top 20 Logistic Regression Coefficients')
    plt.gca().invert_yaxis()
    plt.show()

    return top_coef_df


# Análise

# Verificar se os DataFrames têm as mesmas colunas
def verificar_colunas(df1, df2):
    colunas_df1 = set(df1.columns)
    colunas_df2 = set(df2.columns)
    
    colunas_em_df1_nao_em_df2 = colunas_df1 - colunas_df2
    colunas_em_df2_nao_em_df1 = colunas_df2 - colunas_df1
    
    if colunas_em_df1_nao_em_df2 or colunas_em_df2_nao_em_df1:
        print(f"Colunas em df1 que não estão em df2: {colunas_em_df1_nao_em_df2}")
        print(f"Colunas em df2 que não estão em df1: {colunas_em_df2_nao_em_df1}")
    else:
        print("Os DataFrames têm as mesmas colunas.")

# Analisando as variáveis individualmente e sua relação com a variável 'target'
def analisar_variaveis_numericas(dados, variaveis, target):
    resultados = []
    for var in variaveis:
        descricao = dados[[var]].describe()
        correlacao = dados[[var, target]].corr().iloc[0, 1]
        media_target0 = dados[dados[target] == 0][var].mean()
        media_target1 = dados[dados[target] == 1][var].mean()
        diff_media = media_target1 - media_target0
        perc_missing = dados[var].isna().mean() * 100

        resultados.append({
            'Variavel': var,
            'Media': descricao.loc['mean'][var],
            'Desvio Padrao': descricao.loc['std'][var],
            'Mediana': descricao.loc['50%'][var],
            'Min': descricao.loc['min'][var],
            'Max': descricao.loc['max'][var],
            'Correlacao com Target': correlacao,
            'Media Target 0': media_target0,
            'Media Target 1': media_target1,
            'Diferenca de Media': diff_media,
            'Percentual de Missing': perc_missing
        })
    
    return pd.DataFrame(resultados)



