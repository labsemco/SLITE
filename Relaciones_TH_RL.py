import pandas as pd
import numpy as np
import spacy
import time
from scipy import spatial
import sys
from collections import Counter
from scipy.stats import entropy
import string

#########################    CARGA DE MODELO SPACY NLP Y WORDS EMBEDDINGS     #######################
nlp = spacy.load("en_core_web_md") # modelo de nlp  md

raiz="../ska_RL/"

# Load vectors from dict
def load_vectors_as_dict(path):
    vectors = {}
    with open(path, 'r', encoding="utf8") as f:
        line = f.readline()
        while line:
            # Split on white spaces
            line = line.strip().split(' ')
            if len(line) > 2:
                vectors[line[0]] = np.array([float(l) for l in line[1:]], dtype=np.float32)
            line = f.readline()
    return vectors

# Load vectors in a spacy nlp
def load_vectors_in_lang(nlp, vectors_loc):
    wv= load_vectors_as_dict(vectors_loc)
    nlp.wv = wv

    # # Check if list of oov vectors exists
    # # If so, load, if not, create
    # oov_path,ext = os.path.splitext(vectors_loc)
    # oov_path = oov_path+'.oov.txt'
    # if os.path.exists(oov_path):
    #     nlp.oov = np.loadtxt(oov_path)
    # else:
    fk = list(wv.keys())[0]
    nf = wv[fk].shape[0]
    nlp.oov = np.random.normal(size=(100,nf))
    return 

def get_vector2(w, nlp, nf=300):
    if str(w) in nlp.wv:
        v = nlp.wv[str(w)]
    else: 
        v = np.zeros((1,300))[0]
        #v = np.ones((1,300))[0]
    return v.astype(np.float32)

def get_matrix_rep2(words,nlp, normed=True):
    vecs = np.array([get_vector2(w,nlp) for w in words], dtype=np.float32)
    if len(vecs) == 0:
        vecs = np.ones((1,300), dtype=np.float32)

    # Normalize vectors if desired
    if normed:
        norms = np.linalg.norm(vecs, axis=-1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        vecs /= norms
    return vecs

#ut.load_vectors_in_lang(nlp,"../OPENAI/data/glove.840B.300d.txt") # carga de vectores en nlp.wv
load_vectors_in_lang(nlp,raiz+"data/numberbatch-en-17.04b.txt") # carga de vectores en nlp.wv



##########################################    RELACIONES DE CONCEPTNET    ############################

#cargar relaciones para trabajar de manera local
df_diccionario = pd.read_pickle(raiz+"data/Relaciones_generales.pickle")
df_diccionario_generales = df_diccionario.to_dict()

df_diccionario = pd.read_pickle(raiz+"data/Relaciones_especificas.pickle")
df_diccionario_especificas = df_diccionario.to_dict()

relaciones_gen=["synonym","form_of","is_a","entails","causes","manner_of",
                'has_subevent', 'has_first_subevent', 'has_last_subevent', 
                'has_prerequisite','motivated_by_goal','causes_desire','desires',
                'derived_from']

relaciones_esp=['related_to', 'has_context', 'similar_to', 'etymologically_related_to', 'located_near']

# relaciones_att=[ 'has_a',  'capable_of', 'at_location',  'has_property',  
#             'made_of', 'receives_action', 'created_by']
relaciones_att=[]
##########################################    MÉTODOS NUEVOS      ############################
palabras_negacion_adicionales = [
    "no", "not", "n't", "never", "none", "nobody", "nowhere", "nothing", 
    "neither", "nor", "without", "cannot", "can't", "did not", "didn't", 
    "does not", "doesn't", "do not", "don't", "will not", "won't", 
    "would not", "wouldn't", "could not", "couldn't", "should not", "shouldn't", 
    "must not", "mustn't", "might not", "mightn't", "may not", "lack", 
    "absent", "fail", "deny", "refuse", "against", "opposite", "exclude", 
    "except", "prevent", "avoid", "prohibit", "ban", "restrict", "decline", 
    "reject"]
STOP_WORDS_RT={"'d","'ll","'m","'re","'s","'ve",'a','am','an','and','are','as','at', 'be','i', 'if', 'in', 'is', 'it', 'its',
             'itself','me','my','of','or','our', 'ours', 'ourselves','so', 'than', 'that', 'the','their', 'them',
              'themselves','there', 'thereafter', 'thereby','they','this', 'those','to','thus','us','was', 'we','were',
              'you', 'your', 'yours', 'yourself', 'yourselves', '‘d', '‘ll', '‘m', '‘re', '‘s', '‘ve', '’d', '’ll', '’m',
               '’re', '’s', '’ve'}

def representacion_entidadesDavidSetM(nlp,texto):
    dir_sust=dict()
    palabras=[]
    b=1.0
    pos=[]
    lemmas=[]
    if (type(texto)==type(b) or texto=="" or texto=="n/a" or texto=="nan"):
        return dir_sust,palabras,lemmas,pos
    pos=[]
    lemmas=[]
    tokens=[]
    tokenshead=[]
    tokenschild=[]
    entidades=[]
    doc =nlp(texto.lower())
    for token in doc: # si eres un adjetivo y estas vinculado con un adjetivo deberias de hacer dos tuplas con el NOUN correspondiente
        #print([child for child in token.children],token.text, token.lemma_, token.pos_,token.dep_,token.head.text,token.head.lemma_, token.head.pos_)
        if token.text == "nobody" or token.text == "one":
            if (len(list(token.children))>0):
                for child in token.children:
                    if token.pos_ in ["VERB"]:
                        if child.pos_ not in ["NOUN","VERB","PRON"]:
                            entidades.append((child.lemma_,child.pos_,token.lemma_,token.pos_))
                        else:
                            if token.lemma_ !="be":
                                entidades.append(("","<UKN>",token.lemma_,token.pos_))
                    elif token.pos_ in ["NOUN"]:
                        if child.pos_ not in ["VERB"]:
                            entidades.append((child.lemma_,child.pos_,token.lemma_,token.pos_))
                    else:
                        entidades.append((child.lemma_,child.pos_,token.lemma_,token.pos_))
            else:
                entidades.append(("","<UKN>",token.lemma_,token.pos_))
        elif token.pos_ not in ["DET","ADP","AUX","ADV","ADJ","NUM","PRON"]:
            if (len(list(token.children))>0):
                for child in token.children:
                    if token.pos_ in ["VERB"]:
                        if child.pos_ not in ["NOUN","VERB","PRON"]:
                            entidades.append((child.lemma_,child.pos_,token.lemma_,token.pos_))
                        else:
                            if token.lemma_ !="be":
                                entidades.append(("","<UKN>",token.lemma_,token.pos_))
                    elif token.pos_ in ["NOUN"]:
                        if child.pos_ not in ["VERB"]:
                            entidades.append((child.lemma_,child.pos_,token.lemma_,token.pos_))
                    else:
                        entidades.append(("","<UKN>",token.lemma_,token.pos_))
            else:
                entidades.append(("","<UKN>",token.lemma_,token.pos_))
        elif token.pos_ in ["ADJ"]:
            if (len(list(token.children))>0):
                for child in token.children:
                    if child.pos_ in ["ADP","AUX","ADV"]:
                        entidades.append(("","<UKN>",token.lemma_,token.pos_))
            else:
                if token.head.pos_ in ["ADJ"]:
                    entidades.append((token.lemma_,token.pos_,(token.head).head.lemma_,(token.head).head.pos_))
                else:
                    entidades.append(("","<UKN>",token.lemma_,token.pos_))
        elif token.pos_ in ["ADV"]:
            if (len(list(token.children))==0):
                entidades.append(("","<UKN>",token.lemma_,token.pos_))
        pos.append(token.pos_)
        lemmas.append(token.lemma_)
        tokens.append(token.text)
        #tokenshead.append(token.head.text)
        tokenshead.append((token.head).lemma_)
        tokenschild.append([child for child in token.children])
    print(entidades)
    #print(entidades)
    dir_entidades=dict()
    #  0     1        2      3
    #('', '<UKN>', 'pant', 'NOUN')
    
    for e in entidades:
        #print(e[2])
        if e[3] not in ['PUNCT','CCONJ']:
            if e[2] not in dir_entidades and e[2] not in ["not"]:
                if str(e[0]) in ["no"]:
                    if e[0]!='':
                        dir_entidades[str(e[2])]=set([e[0]])
                elif e[1] in ["<UKN>","DET","ADP",'CCONJ','PRON']:
                    if e[2] in ['is']:
                        dir_entidades["be"]=set()
                    else:
                        dir_entidades[e[2]]=set()
                else:
                    # if e[1] in ["NOUN"]:
                    #     dir_entidades[str(e[2])+" "+str(e[0])]=""
                    if e[1] in ["NOUN"]:
                        if e[0] not in dir_entidades:
                            dir_entidades[str(e[0])]=set()
                        if e[2] not in dir_entidades:
                            dir_entidades[str(e[2])]=set()
                    else:
                        if e[1] not in ["PRON","PUNCT"]:# segundo agregue
                            if e[0]!='':
                                dir_entidades[e[2]]=set([e[0]])
                            elif e[1] in ["<UKN>"] and e[3] in ['NOUN']:
                                if e[0]=='':
                                    dir_entidades[e[2]]=set()
            else:
                if e[2] not in ["not"]: #checar
                    if str(e[0]) in ["no"]:
                        sa=dir_entidades[e[2]]
                        if e[0]!='':
                            sa.add(e[0])
                            dir_entidades[str(e[2])]=sa
                    # elif e[1] in ["NOUN"]:
                    #         dir_entidades[str(e[2])+" "+str(e[0])]=""
                    elif e[1] in ["NOUN"]:
                        if e[0] not in dir_entidades:
                            dir_entidades[str(e[0])]=set()
                        if e[2] not in dir_entidades:
                            dir_entidades[str(e[2])]=set()
                    #elif e[1] not in ["<UNK>","DET","ADP",'CCONJ','PRON',"PUNCT"]:
                    elif e[1] not in ["<UKN>","DET",'CCONJ','PRON',"PUNCT"]:
                        if len(dir_entidades[str(e[2])])==0:
                            if e[0]!='':
                                dir_entidades[str(e[2])]=set([e[0]])
                        else:
                            sa=dir_entidades[e[2]]
                            if e[0]!='':
                                sa.add(e[0])
                                dir_entidades[str(e[2])]=sa
    print("pos ",pos)
    print("lemas ",lemmas)
    print("text ",tokens)
    print("head ",tokenshead)
    print("child ",tokenschild)
    if len(list(dir_entidades.keys()))==0:
        lemmas=[]
        pos=[]
        for token in doc: 
            if token.pos_ in ["ADP","VERB","ADV","ADJ","NUM","NOUN"]:
                dir_entidades[token.lemma_]=set()
                lemmas.append(token.lemma_)
                pos.append(token.pos_)
        return dir_entidades,list(dir_entidades.keys()),lemmas,pos
        # ls=texto.split()
        # for a in ls:
        #     dir_entidades[a]=set()
        # return dir_entidades,ls,lemmas,pos
    else:        
        return dir_entidades,list(dir_entidades.keys()),lemmas,pos

def eliminacion_espacios(lista):
    eliminar_espacios=lista.count("")
    if eliminar_espacios>0:
        for espacios in range(eliminar_espacios):
            lista.remove("")
    eliminar_espacios=lista.count("be")
    if eliminar_espacios>0:
        for espacios in range(eliminar_espacios):
            lista.remove("be")
    return lista

def found_neg(at_t):
    for t in at_t:
        if t in palabras_negacion_adicionales:
            return True,t
    return False,""

def check_entail_syn(wt,wh):
    # guardo conjuntos de sinonimos para un uso posterior
    if wt ==wh:                #COMPARACION DIRECTA DE wt y wh
        return True,"same"
    
    synt_i=set()
    if wt in df_diccionario_generales:
        synt_i=synt_i.union(df_diccionario_generales[wt]["synonym"])
        synt_i=synt_i.union(df_diccionario_generales[wt]["form_of"])
        synt_i=synt_i.union(df_diccionario_generales[wt]["defined_as"])
    if wt in df_diccionario_especificas:
        synt_i=synt_i.union(df_diccionario_especificas[wt]["synonym"])
    
    synh_i=set()
    if wh in df_diccionario_generales:
        synh_i=synh_i.union(df_diccionario_generales[wh]["synonym"])
        synh_i=synh_i.union(df_diccionario_generales[wh]["form_of"])
        synh_i=synh_i.union(df_diccionario_generales[wh]["defined_as"])
    if wh in df_diccionario_especificas:
        synh_i=synh_i.union(df_diccionario_especificas[wh]["synonym"])
       
    if wh in synt_i:
        return True,"synonym"
    elif wt in synh_i:
        return True,"synonym"
    elif len(synt_i.intersection(synh_i))>0:
        return True,"synonym"
    else:
        return False,""
         
def check_entail_gen(wt,wh): # t->h 
    synh_i=set()
    if wh in df_diccionario_especificas:
        synh_i=synh_i.union(df_diccionario_especificas[wh]["synonym"])
        for r_g in relaciones_gen:
            synh_i=synh_i.union(df_diccionario_especificas[wh][r_g])
    if wt in synh_i:
        if wt not in df_diccionario_generales[wh]["antonym"] and wt not in df_diccionario_especificas[wh]["antonym"] and wt not in df_diccionario_generales[wh]["distinct_from"] and wt not in df_diccionario_especificas[wh]["distinct_from"]:
            return True,"general"
        else:
            return False,""
    if wt in df_diccionario_generales:
        for r_g in relaciones_gen:
            if len(synh_i.intersection(df_diccionario_generales[wt][r_g]))>0 and wt not in df_diccionario_generales[wh]["antonym"] and wt not in df_diccionario_especificas[wh]["antonym"] and wt not in df_diccionario_generales[wh]["distinct_from"] and wt not in df_diccionario_especificas[wh]["distinct_from"]:
                #print(wt,wh,r_g,"generales",len(synh_i.intersection(df_diccionario_generales[wt][r_g])))
                return True,r_g
        return False,""  # hiponimos
    else:
        return False,""

def check_contradiction_ant(wt,wh):
    relaciones_g2=["antonym","distinct_from"]
    # guardo conjuntos de sinonimos para un uso posterior
    synt_i=set()
    if wt in df_diccionario_generales:
        synt_i=synt_i.union(df_diccionario_generales[wt]["synonym"])
        synt_i=synt_i.union(df_diccionario_generales[wt]["form_of"])
        synt_i=synt_i.union(df_diccionario_generales[wt]["defined_as"])
    if wt in df_diccionario_especificas:
        synt_i=synt_i.union(df_diccionario_especificas[wt]["synonym"])
    
    synh_i=set()
    if wh in df_diccionario_generales:
        synh_i=synh_i.union(df_diccionario_generales[wh]["synonym"])
        synh_i=synh_i.union(df_diccionario_generales[wh]["form_of"])
        synh_i=synh_i.union(df_diccionario_generales[wh]["defined_as"])
    if wh in df_diccionario_especificas:
        synh_i=synh_i.union(df_diccionario_especificas[wh]["synonym"])

    antt_i=set()
    if wt in df_diccionario_generales:
        antt_i=antt_i.union(df_diccionario_generales[wt]["antonym"])
        antt_i=antt_i.union(df_diccionario_generales[wt]["distinct_from"])
    if wt in df_diccionario_especificas:
        antt_i=antt_i.union(df_diccionario_especificas[wt]["antonym"])
        antt_i=antt_i.union(df_diccionario_especificas[wt]["distinct_from"])
    anth_i=set()
    if wh in df_diccionario_generales:
        anth_i=anth_i.union(df_diccionario_generales[wh]["antonym"])
        anth_i=anth_i.union(df_diccionario_generales[wh]["distinct_from"])
    if wh in df_diccionario_especificas:
        anth_i=anth_i.union(df_diccionario_especificas[wh]["antonym"])
        anth_i=anth_i.union(df_diccionario_especificas[wh]["distinct_from"])

    if wh in antt_i or wt in anth_i:
        return True,"antonym"
    elif len(synh_i.intersection(antt_i))>0 or len(synt_i.intersection(anth_i))>0:
        return True,"antonym"

    # check antonimos
    # for r_g in relaciones_g2:
    #     if wh in df_diccionario_generales[wt][r_g] or len(synh_i.intersection(df_diccionario_generales[wt][r_g]))>0:
    #         return True,r_g
    #     elif wh in df_diccionario_especificas[wt][r_g] or len(synh_i.intersection(df_diccionario_especificas[wt][r_g]))>0:
    #         return True,r_g
    #     elif wt in df_diccionario_generales[wh][r_g] or len(synt_i.intersection(df_diccionario_generales[wh][r_g]))>0:
    #         return True,r_g
    #     elif wt in df_diccionario_especificas[wh][r_g] or len(synt_i.intersection(df_diccionario_especificas[wh][r_g]))>0:
    #         return True,r_g   
    return False,""

def check_info_adicional(wt,wh):
    setT_ia=set(wt)
    setH_ia=set(wh)
    #temSetT=set()
    if wt in df_diccionario_generales:            
        for r_g in relaciones_att:
            setT_ia=setT_ia.union(df_diccionario_generales[wt][r_g])
    if wh in df_diccionario_generales:
        for r_g in relaciones_att:
            setH_ia=setH_ia.union(df_diccionario_generales[wh][r_g])

# # extendemos 
#     for r_g in ["part_of"]:
#         for e in setT_ia:
#             if e in df_diccionario_generales:
#                 temSetT=temSetT.union(df_diccionario_generales[e][r_g])
#     setT_ia=setT_ia.union(temSetT)
#     #print("texto",setT_ia)
#     #print("hipotesis",setH_ia)
    
    if len(setT_ia.intersection(setH_ia))>0:
        #print("inter",intersec)
        return True,"properties"
    return False,""
    
# def check_neutral_rel(wt,wh):
#     #relaciones_g3=["related_to","similar_to"]
#     if wh in df_diccionario_generales:
#         synh_i=df_diccionario_especificas[wh]["synonym"]
#         synh_i=synh_i.union(df_diccionario_generales[wh]["synonym"])
#         synh_i=synh_i.union(df_diccionario_generales[wh]["form_of"])
#         synh_i=synh_i.union(df_diccionario_generales[wh]["is_a"])
#         synh_i=synh_i.union(df_diccionario_generales[wh]["manner_of"])
#         synh_i=synh_i.union(df_diccionario_generales[wh]["used_for"])
#     else:
#         synh_i=set()
#     if wt in df_diccionario_generales and wh in df_diccionario_generales:
#         for r_g in ["form_of","is_a","used_for","entails","causes","synonym","manner_of"]:
#             if wh in df_diccionario_especificas[wt][r_g]:
#                 return True,r_g
#             if len(synh_i.intersection(df_diccionario_especificas[wt][r_g]))>0:
#                 return True,r_g
#         # for r_g in relaciones_g3:
#         #     if wh in df_diccionario_generales[wt][r_g]:
#         #         return True,r_g
#         # for r_g in relaciones_g3:
#         #     if wh in df_diccionario_especificas[wt][r_g]:
#         #         return True,r_g
#         return False,""
#     else:
#         return False,""

def get_atributos(at_t,at_h):
    #t_temp = set(eliminacion_espacios(at_t.split(",")))
    #h_temp = set(eliminacion_espacios(at_h.split(",")))
    t_temp = at_t
    h_temp = at_h

    t_atributos =set()
    h_atributos =set()
    
    for t_ in t_temp:
        if t_ not in STOP_WORDS_RT and t_!="" and t_!=" " and t_!=",":
            t_atributos.add(t_)
    for h_ in h_temp:
        if h_ not in STOP_WORDS_RT and h_!="" and h_!=" " and h_!=",":
            h_atributos.add(h_)
    return " ".join(t_atributos)," ".join(h_atributos)

def check_atributos(at_t,at_h):
    #t_temp = set(eliminacion_espacios(at_t.split(",")))
    #h_temp = set(eliminacion_espacios(at_h.split(",")))
    t_temp = at_t
    h_temp = at_h
    
    t_atributos =set()
    h_atributos =set()
    
    for t_ in t_temp:
        if t_ not in STOP_WORDS_RT and t_!="" and t_!=" " and t_!=",":
            t_atributos.add(t_)
    for h_ in h_temp:
        if h_ not in STOP_WORDS_RT and h_!="" and h_!=" " and h_!=",":
            h_atributos.add(h_)
    print("atributos de T",t_atributos)
    print("atributos de H",h_atributos)
    vt,fN_t=found_neg(t_atributos)
    vh,fN_h=found_neg(h_atributos)
    if vt!=False:
        #return False,fN_t,"",0
        return False," ".join(t_atributos)," ".join(h_atributos),0
    if vh!=False:
        return False," ".join(t_atributos)," ".join(h_atributos),0
        #return False,"",fN_h,0
    # Checar cuantos atributos de h están contenidos en T
    found_att_t=[]
    found_att_h=[]
    matches=0
    for h_a in h_atributos:
        for t_a in t_atributos:
            # relaciones generales
            if h_a in t_atributos:
                matches+=1
                found_att_t.append(t_a)
                found_att_h.append(h_a)
                break
            verificacion,tupla=check_entail_syn(t_a,h_a) #COMPARACION SINONIMOS DE wt y wh
            if verificacion:
                matches+=1
                found_att_t.append(t_a)
                found_att_h.append(h_a)
                break
            # checo en relaciones generales si se encuentra el token de la hipótesis
            verificacion,tupla=check_entail_gen(t_a,h_a) #COMPARACION GENERAL DE wt y wh
            if verificacion:
                found_att_t.append(t_a)
                found_att_h.append(h_a)
                matches+=1
                break
            # contradicciones
            verificacion,tupla=check_contradiction_ant(t_a,h_a) #COMPARACION CONTRA DE wt y wh
            if verificacion:
                return False,t_a,h_a,0
            #conjuntos de información adicional part_of escalar y subir
            # verificacion,tupla=check_info_adicional(t_a,h_a) #COMPARACION INFO ADICIONAL DE wt y wh
            # if verificacion:
            #     found_att_t.append(t_a)
            #     found_att_h.append(h_a)
            #     matches+=1
            #     break
            #especificas
            # verificacion,tupla=check_neutral_rel(t_a,h_a) #COMPARACION NEUTRAL DE wt y wh
            # if verificacion:
            #     return False,t_a,h_a,2
            verificacion,tupla=check_entail_gen(h_a,t_a) #COMPARACION NEUTRAL DE wt y wh cambiamos el orden
            if verificacion:
                return False,t_a,h_a,2
    if len(h_atributos)==0:
        return True, " ".join(t_atributos),"",1
    elif len(t_atributos)==0 and len(h_atributos)==0:
        return True, "","",1
    elif matches==len(h_atributos):
        return True, " ".join(found_att_t)," ".join(found_att_h),1
    elif len(t_atributos)==0 and len(h_atributos)>0:
        return False,""," ".join(h_atributos),2
    elif matches!=len(h_atributos):
        return False," ".join(t_atributos)," ".join(h_atributos),2
    else:
        return False, " ".join(t_atributos)," ".join(h_atributos),0

def entropia(X):
    """Devuelve el valor de entropia de una muestra de datos""" 
    probs = [np.mean(X == valor) for valor in set(X)]
    return round(sum(-p * np.log2(p) for p in probs), 3)


def jaro_distance(s1, s2,sinT,sinH,HipT,hipH):
    #print(s1, s2,sinT,sinH,HipT,hipH)
    bandera=True

    # Length of two strings
    len1 = len(s1)
    len2 = len(s2)

    # If the listas de tokens are equal 
    if len1==len2:
        for i in range(len1):
            if s1[i]!=s2[i]:
                bandera=False
                break
        if (bandera):
            return 1.0,1.0; 
 
    if (len1 == 0 or len2 == 0) :
        return 0.0,0.0; 
 
    # Maximum distance upto which matching 
    # is allowed 
    max_dist = (max(len(s1), len(s2)) // 2 )-1 ; 
 
    # Count of matches 
    match = 0; 
 
    # Hash for matches 
    hash_s1 = [0] * len(s1)
    hash_s2 = [0] * len(s2)
 
    # Traverse through the first string 
    for i in range(len1):
 
        # Check if there is any matches
        for j in range(max(0, i - max_dist), 
                       min(len2, i + max_dist + 1)):
            #print(s1[i],s2[j])
            # If there is a match or is contain in a bag of sinomys of tk
            if ((s1[i] == s2[j] or s1[i] in sinH[j] or s2[j] in sinT[i]) and hash_s2[j] == 0) : 
                if (i>0 and s1[i-1] not in ["no","not","n't"]) and (j>0 and s2[j-1] not in ["no","not","n't"]):
                    print(s1[i],s2[j],"sinonimos")
                    hash_s1[i] += 1; 
                    hash_s2[j] += 1; 
                    match += 1; 
                    break
                elif (i>0 and s1[i-1] not in ["no","not","n't"] and j==0):
                    print(s1[i],s2[j],"sinonimos")
                    hash_s1[i] += 1; 
                    hash_s2[j] += 1; 
                    match += 1; 
                    break
                elif (j>0 and s2[j-1] not in ["no","not","n't"] and i==0):
                    print(s1[i],s2[j],"sinonimos")
                    hash_s1[i] += 1; 
                    hash_s2[j] += 1; 
                    match += 1; 
                    break

            elif ((s1[i] in hipH[j] or len((sinT[i]).intersection(hipH[j]))>0) and hash_s2[j] == 0):
                if (i>0 and s1[i-1] not in ["no","not","n't"]) or (j>0 and s2[j-1] not in ["no","not","n't"]):
                    print("hiponimos",s2[j],s1[i])#,(sinT[i]).intersection(hipH[j]))
                    hash_s1[i] += 1; 
                    hash_s2[j] += 1; 
                    match += 1; 
                    break
                elif (i>0 and s1[i-1] not in ["no","not","n't"] and j==0):
                    print("hiponimos",s2[j],s1[i])#,(sinT[i]).intersection(hipH[j]))
                    hash_s1[i] += 1; 
                    hash_s2[j] += 1; 
                    match += 1; 
                    break
                elif (j>0 and s2[j-1] not in ["no","not","n't"] and i==0):
                    print("hiponimos",s2[j],s1[i])#,(sinT[i]).intersection(hipH[j]))
                    hash_s1[i] += 1; 
                    hash_s2[j] += 1; 
                    match += 1; 
                    break
            elif ((s2[j] in HipT[i] or len((sinH[j]).intersection(HipT[i]))>0) and hash_s2[j] == 0): 
                if (i>0 and s1[i-1] not in ["no","not","n't"]) or (j>0 and s2[j-1] not in ["no","not","n't"]):
                    print("hiperonimos sobre sinonimos",s2[j],s1[i])
                    hash_s1[i] += 1; 
                    hash_s2[j] += 1; 
                    match += 1; 
                    break
                elif (i>0 and s1[i-1] not in ["no","not","n't"] and j==0):
                    print("hiperonimos sobre sinonimos",s2[j],s1[i])
                    hash_s1[i] += 1; 
                    hash_s2[j] += 1; 
                    match += 1; 
                    break
                elif (j>0 and s2[j-1] not in ["no","not","n't"] and i==0):
                    print("hiperonimos sobre sinonimos",s2[j],s1[i])
                    hash_s1[i] += 1; 
                    hash_s2[j] += 1; 
                    match += 1; 
                    break
            # elif len((hipH[j]).intersection(HipT[i]))>0 and hash_s2[j] == 0: 
            #     print("hiperonimos3",s2[j],s1[i],(hipH[j]).intersection(HipT[i]))
            #     hash_s1[i] += 1; 
            #     hash_s2[j] += 1; 
            #     match += 1; 
            #     break
            
    print(hash_s1)
    print(hash_s2)
    print(match)
    # If there is no match 
    if (match == 0) :
        return 0.0,0.0; 
 
    # Number of transpositions 
    t = 0; 
 
    point = 0; 
 
    # Count number of occurrences 
    # where two characters match but 
    # there is a third matched character 
    # in between the indices 
    for i in range(len1) : 
        if (hash_s1[i]) :
            # Find the next matched character 
            # in second string 
            while (hash_s2[point] == 0) :
                point += 1; 
 
            if (s1[i] != s2[point]) :
                point += 1
                t += 1
            else :
                point += 1    
    t /= 2; 
    #Return the Jaro Similarity 
    return match / len2,(( match / len2  + match / len1 +
            (match - t) / match ) / 3.0); 

def jaro_distance_semantic(s1, s2, sinT, sinH, HipT, hipH):
    len1, len2 = len(s1), len(s2)

    if len1 == 0 or len2 == 0:
        return 0.0, 0.0

    max_dist = max(len1, len2) // 2 
    match = 0
    hash_s1 = [0] * len1
    hash_s2 = [0] * len2

    # Fase de Coincidencias (Matches)
    for i in range(len1):
        for j in range(max(0, i - max_dist), min(len2, i + max_dist + 1)):
            if hash_s2[j] == 0:
                # 1. Sinonimia
                es_sinonimo = (s1[i] == s2[j] or s1[i] in sinH[j] or s2[j] in sinT[i])
                
                # 2. Hiponimia
                es_hiponimo = (s1[i] in hipH[j] or len(sinT[i].intersection(hipH[j])) > 0)
                
                # 3. Hiperonimia
                es_hiperonimo = (s2[j] in HipT[i] or len(sinH[j].intersection(HipT[i])) > 0)

                if es_sinonimo or es_hiponimo or es_hiperonimo:
                    # # --- NUEVA LÓGICA DE NEGACIÓN UNIFICADA ---
                    # # Un token es válido si no tiene una negación antes
                    t1_valido = (i == 0 or s1[i-1] not in palabras_negacion_adicionales)
                    t2_valido = (j == 0 or s2[j-1] not in palabras_negacion_adicionales)
                    # # Solo evaluamos semántica si AMBOS están en el mismo estado (no negados)
                    # if t1_valido and t2_valido:
                    #     hash_s1[i] = 1
                    #     hash_s2[j] = 1
                    #     match += 1
                    #     break

                    # El cambio clave: deben tener el mismo estado lógico
                    if t1_valido == t2_valido:
                        hash_s1[i] = 1
                        hash_s2[j] = 1
                        match += 1
                        
                        # Opcional: Si ambos están negados, también deberíamos 
                        # intentar marcar la palabra "no" anterior como match para no penalizar el score
                        if not t1_valido and i > 0 and j > 0:
                            if s1[i-1] == s2[j-1] and hash_s2[j-1] == 0:
                                hash_s1[i-1] = 1
                                hash_s2[j-1] = 1
                                match += 1
                        break

    if match == 0:
        return 0.0, 0.0

    # Fase de Transposiciones
    t = 0
    point = 0
    for i in range(len1):
        if hash_s1[i]:
            while hash_s2[point] == 0:
                point += 1
            
            # Para las transposiciones, comparamos si el "concepto" es el mismo
            # considerando que ya pasaron el filtro de negación arriba
            match_exacto = (s1[i] == s2[point])
            match_semantico = (s1[i] in sinH[point] or s2[point] in sinT[i] or s1[i] in hipH[point])
            
            if not (match_exacto or match_semantico):
                t += 1
            point += 1

    t /= 2.0
    
    # Cálculo de Similitud Jaro
    sim_jaro = (match/len1 + match/len2 + (match - t)/match) / 3.0
    
    return match / max(len1, len2), sim_jaro

def jaro_distance_semantic_final(s1, s2, sinT, sinH, HipT, hipH, antT, antH, palabras_negacion):
    len1, len2 = len(s1), len(s2)

    if len1 == 0 or len2 == 0:
        return 0.0, 0.0

    max_dist = max(len1, len2) // 2 
    match = 0
    penalizacion = 0 # <--- Contador de contradicciones lógicas
    hash_s1 = [0] * len1
    hash_s2 = [0] * len2

    # Fase de Coincidencias (Matches)
    for i in range(len1):
        for j in range(max(0, i - max_dist), min(len2, i + max_dist + 1)):
            if hash_s2[j] == 0:
                # 1. Relaciones semánticas
                es_sin = (s1[i] == s2[j] or s1[i] in sinH[j] or s2[j] in sinT[i])
                es_hip = (s1[i] in hipH[j] or len(sinT[i].intersection(hipH[j])) > 0)
                es_hiper = (s2[j] in HipT[i] or len(sinH[j].intersection(HipT[i])) > 0)
                es_ant = (s1[i] in antH[j] or s2[j] in antT[i])

                # 2. Estados lógicos (True = Afirmativo, False = Negado)
                t1_valido = (i == 0 or s1[i-1] not in palabras_negacion)
                t2_valido = (j == 0 or s2[j-1] not in palabras_negacion)

                # --- LÓGICA DE MATCH (EQUIVALENCIA) ---
                # Caso A: Son parecidos y tienen el mismo estado (Ej: "es auto" vs "es coche")
                # Caso B: Son opuestos y tienen estado distinto (Ej: "no es caro" vs "es barato")
                if ((es_sin or es_hip or es_hiper) and (t1_valido == t2_valido)) or \
                   (es_ant and (t1_valido != t2_valido)):
                    
                    hash_s1[i], hash_s2[j] = 1, 1
                    match += 1
                    
                    # Match de la partícula "no" si ambos están negados
                    if not t1_valido and i > 0 and j > 0:
                        if s1[i-1] == s2[j-1] and hash_s2[j-1] == 0:
                            hash_s1[i-1], hash_s2[j-1] = 1, 1
                            match += 1
                    break

                # --- LÓGICA DE PENALIZACIÓN (CONTRADICCIÓN) ---
                # Caso A: Son parecidos pero uno está negado (Ej: "pagó" vs "no pagó")
                # Caso B: Son opuestos pero ambos están afirmados (Ej: "ganar" vs "perder")
                elif ((es_sin or es_hip or es_hiper) and (t1_valido != t2_valido)) or \
                     (es_ant and (t1_valido == t2_valido)):
                    
                    # Marcamos como usados e incrementamos penalización
                    hash_s1[i], hash_s2[j] = 1, 1
                    penalizacion += 1.5 # Un valor > 1 asegura que la similitud caiga rápido
                    break

    # Aplicamos el castigo al conteo de matches
    match_final = max(0, match - penalizacion)

    if match_final == 0:
        return 0.0, 0.0

    # Fase de Transposiciones (usando el match_final)
    t = 0
    point = 0
    for i in range(len1):
        if hash_s1[i]:
            while point < len2 and hash_s2[point] == 0:
                point += 1
            
            if point < len2:
                match_exacto = (s1[i] == s2[point])
                match_sem = (s1[i] in sinH[point] or s2[point] in sinT[i] or s1[i] in hipH[point])
                if not (match_exacto or match_sem):
                    t += 1
                point += 1

    t /= 2.0
    
    # Cálculo de Similitud Jaro con matches penalizados
    sim_jaro = (match_final/len1 + match_final/len2 + (match_final - t)/match_final) / 3.0
    
    return match_final / max(len1, len2), max(0, sim_jaro)

# Transfer Entropy
def calculate_te(source, target):
    """Calcula la Transfer Entropy entre dos secuencias discretizadas."""
    if len(source) < 2: return 0
    x_curr, x_past, y_past = target[1:], target[:-1], source[:-1]
    N = len(x_curr)
    
    c_3 = Counter(zip(x_curr, x_past, y_past))
    c_2_xp_yp = Counter(zip(x_past, y_past))
    c_2_xn_xp = Counter(zip(x_curr, x_past))
    c_1_xp = Counter(x_past)
    
    te = 0.0
    for (xn, xp, yp), count in c_3.items():
        p_joint = count / N
        p_cond_full = count / c_2_xp_yp[(xp, yp)]
        p_cond_limited = c_2_xn_xp[(xn, xp)] / c_1_xp[xp]
        te += p_joint * np.log2(p_cond_full / p_cond_limited)
    return te

def calculate_te_safe(source, target, delay=1):
    """Calcula la Transfer Entropy con validación de longitud."""
    # Sincronizamos longitudes
    min_len = min(len(source), len(target))
    source, target = source[:min_len], target[:min_len]
    
    # El tamaño efectivo para calcular transiciones
    N = len(target) - delay
    
    # VALIDACIÓN: Si no hay suficientes datos para ver pasado y presente, TE = 0
    if N <= 0:
        return 0.0

    x_curr = target[delay:]
    x_past = target[:-delay]
    y_past = source[:-delay]
    
    c_3 = Counter(zip(x_curr, x_past, y_past))
    c_2_xp_yp = Counter(zip(x_past, y_past))
    c_2_xn_xp = Counter(zip(x_curr, x_past))
    c_1_xp = Counter(x_past)
    
    te = 0.0
    for (xn, xp, yp), count in c_3.items():
        p_joint = count / N
        p_cond_full = count / c_2_xp_yp[(xp, yp)]
        p_cond_limited = c_2_xn_xp[(xn, xp)] / c_1_xp[xp]
        
        # Evitamos log(0)
        if p_cond_full > 0 and p_cond_limited > 0:
            te += p_joint * np.log2(p_cond_full / p_cond_limited)
    return te

def analyze_h_reduction(df, remove_cols):
    """
    Analiza cómo afecta eliminar conceptos de la Hipótesis (columnas)
    manteniendo la Premisa intacta.
    """
    df_disc = df.round(1)
    
    # TE Original
    te_orig = calculate_te(df_disc.values.flatten(), df_disc.T.values.flatten())
    
    # Eliminamos solo COLUMNAS (Hipótesis)
    df_filtered = df_disc.drop(columns=remove_cols, errors='ignore')
    
    # TE Post-filtrado
    # Nota: El target_seq ahora es más corto
    te_filt = calculate_te(df_filtered.values.flatten(), df_filtered.T.values.flatten())
    
    return te_orig, te_filt
def find_key_informant_words(df):
    """
    Calcula la contribución de Transfer Entropy de cada palabra 
    de la hipótesis por separado.
    """
    df_disc = df.round(1)
    results = {}
    
    # La fuente siempre es la premisa completa (todas las filas aplanadas)
    source_seq = df_disc.values.flatten()
    
    # Iteramos sobre cada columna (palabra de la hipótesis)
    for col in df_disc.columns:
        # La secuencia destino es solo esta palabra
        target_seq = df_disc[col].values
        
        # Calculamos la TE para esta palabra específica
        te_val = calculate_te_safe(source_seq, target_seq)
        results[col] = te_val
    
    # Ordenamos de mayor a mayor contribución
    ranking = dict(sorted(results.items(), key=lambda item: item[1], reverse=True))
    return ranking

def jaro_rte_expert(T, H, sinT, sinH, antT, antH, HipT, hipH, palabras_negacion):
    len_t, len_h = len(T), len(H)
    if len_t == 0 or len_h == 0: return 0.0, 0.0

    max_dist = max(len_t, len_h) // 2
    match = 0
    invalido_por_contradiccion = False
    
    hash_t = [0] * len_t
    hash_h = [0] * len_h

    for i in range(len_t):
        for j in range(max(0, i - max_dist), min(len_h, i + max_dist + 1)):
            if hash_h[j] == 0:
                # 1. Análisis de Polaridad (Negación)
                t_afirmativo = (i == 0 or T[i-1] not in palabras_negacion)
                h_afirmativo = (j == 0 or H[j-1] not in palabras_negacion)
                mismo_estado = (t_afirmativo == h_afirmativo)

                # 2. Evaluación de Relaciones Semánticas
                es_sin = (T[i] == H[j] or H[j] in sinT[i] or T[i] in sinH[j])
                # Para RTE: H debe ser más general que T (Hiperónimo de T)
                es_gen = (H[j] in HipT[i]) 
                es_ant = (H[j] in antT[i] or T[i] in antH[j])
                #es_hip = (T[i] in hipH[j] or sinT[i] in antH[j])

                # --- LÓGICA DE INFERENCIA (MATCHES) ---
                if mismo_estado:
                    if es_sin or es_gen:
                        match += 1
                        hash_t[i], hash_h[j] = 1, 1
                        break
                    elif es_ant:
                        # Afirmar un antónimo (T: "está vivo", H: "está muerto")
                        invalido_por_contradiccion = True
                else:
                    # Estados diferentes (uno negado, uno afirmativo)
                    if es_sin or es_gen:
                        # T: "está vivo", H: "no está vivo" -> Contradicción
                        invalido_por_contradiccion = True
                    elif es_ant:
                        # T: "no está vivo", H: "está muerto" -> Entailment
                        match += 1
                        hash_t[i], hash_h[j] = 1, 1
                        break

    if match == 0: return 0.0, 0.0

    # 3. Transposiciones (con sensibilidad semántica)
    t = 0
    p = 0
    for i in range(len_t):
        if hash_t[i]:
            while hash_h[p] == 0: p += 1
            # Si no coinciden ni por texto, ni por tus conjuntos, es transposición
            if T[i] != H[p] and H[p] not in sinT[i] and H[p] not in HipT[i]:
                t += 1
            p += 1
    t /= 2.0

    # 4. Score Final
    score_jaro = (match/len_t + match/len_h + (match-t)/match) / 3.0
    
    # Aplicar penalización "Kill-switch" si se detectó contradicción
    if invalido_por_contradiccion:
        score_jaro *= 0.01 

    return match / len_h, score_jaro

def jaro_torres_rte_expert(T, H, sinT, sinH, antT, antH, HipT, HipH, hipT, hipH, palabras_negacion):
    len_t, len_h = len(T), len(H)
    if len_t == 0 or len_h == 0: return 0.0, 0.0

    max_dist = max(len_t, len_h) // 2
    match = 0
    penalizado_por_especificidad = False
    invalido_por_contradiccion = False
    
    hash_t = [0] * len_t
    hash_h = [0] * len_h

    for i in range(len_t):
        for j in range(max(0, i - max_dist), min(len_h, i + max_dist + 1)):
            if hash_h[j] == 0:
                # 1. Análisis de Polaridad (Negación)
                t_afirmativo = (i == 0 or T[i-1] not in palabras_negacion)
                h_afirmativo = (j == 0 or H[j-1] not in palabras_negacion)
                mismo_estado = (t_afirmativo == h_afirmativo)

                # 2. Evaluación de Relaciones Semánticas
                es_sin = (T[i] == H[j] or H[j] in sinT[i] or T[i] in sinH[j]
                          or sinT[i].intersection(sinH[j]))

                # Para RTE: H debe ser más general que T (Hiperónimo de T)
                es_gen = (H[j] in HipT[i] or T[i] in hipH[j]
                          or sinT[i].intersection(hipH[j]) 
                          or sinH[j].intersection(HipT[i]))
                # Para RTE: H debe ser más general que T (Hiperónimo de T)
                es_esp = (T[i] in HipH[j] or H[j] in hipT[i]
                          or sinT[i].intersection(HipH[j]) 
                          or sinH[j].intersection(hipT[i]))

                es_ant = (H[j] in antT[i] or T[i] in antH[j] 
                          or sinH[j].intersection(antT[i]) 
                          or sinT[i].intersection(antH[j]))

                # --- LÓGICA DE INFERENCIA (MATCHES) ---
                if mismo_estado:
                    if es_sin or es_gen:
                        match += 1
                        hash_t[i], hash_h[j] = 1, 1
                        break
                    elif es_esp:
                        # No sumamos match completo o marcamos para penalizar
                        penalizado_por_especificidad = True
                    elif es_ant:
                        # Afirmar un antónimo (T: "está vivo", H: "está muerto")
                        invalido_por_contradiccion = True
                else:
                    # Estados diferentes (uno negado, uno afirmativo)
                    if es_sin or es_gen or es_esp:
                        # T: "está vivo", H: "no está vivo" -> Contradicción
                        invalido_por_contradiccion = True
                    elif es_ant:
                        # T: "no está vivo", H: "está muerto" -> Entailment
                        match += 1
                        hash_t[i], hash_h[j] = 1, 1
                        break

    if match == 0: return 0.0, 0.0

    # 3. Transposiciones (con sensibilidad semántica coherente)
    t = 0
    p = 0
    for i in range(len_t):
        if hash_t[i]:
            # Buscamos la siguiente palabra en H que tuvo un match
            while p < len_h and hash_h[p] == 0: 
                p += 1
            
            if p < len_h:
                # --- LA CLAVE ESTÁ AQUÍ ---
                # Solo es transposición si NO hay ninguna relación de equivalencia o jerarquía válida
                es_equivalente = (T[i] == H[p] or 
                                H[p] in sinT[i] or 
                                T[i] in sinH[p] or 
                                sinT[i].intersection(sinH[p]))
                
                es_jerarquia_valida = (H[p] in HipT[i] or 
                                    T[i] in hipH[p] or 
                                    sinT[i].intersection(hipH[p]) or 
                                    sinH[p].intersection(HipT[i]))

                if not (es_equivalente or es_jerarquia_valida):
                    t += 1
                p += 1
    t /= 2.0

    # --- NUEVA SECCIÓN: PENALIZACIÓN POR INFORMACIÓN EXTRA (NEUTRAL) ---
    # Contamos cuántas palabras de la hipótesis no tuvieron pareja (hash_h[j] == 0)
    h_huerfanas = sum(1 for matched in hash_h if matched == 0)
    
    # Calculamos el ratio de "novedad" en H
    ratio_info_nueva = h_huerfanas / len_h if len_h > 0 else 0
    
    # --- FIN DE LA NUEVA SECCIÓN ---

    # 4. Score Final
    score_jaro = (match/len_t + match/len_h + (match-t)/match) / 3.0
    
    # Aplicar penalización "Kill-switch" si se detectó contradicción
    if invalido_por_contradiccion:
        score_jaro *= 0.01 
    elif penalizado_por_especificidad:
        score_jaro *= 0.6   # Penalización moderada (Falta de especificidad en T)

    # Si H tiene mucha información que T no mencionó, castigamos el score
    # para empujarlo hacia la zona de 'NEUTRAL' en tu gráfica.
    if ratio_info_nueva > 0.1: # Umbral ajustable (25% de info nueva)
        score_jaro *= (1.0 - (ratio_info_nueva * 0.4))
        
    return match / len_h, score_jaro

def jaro_torres_rte_expert2(T, H, sinT, sinH, antT, antH, HipT, HipH, hipT, hipH, palabras_negacion):
    len_t, len_h = len(T), len(H)
    if len_t == 0 or len_h == 0: return 0.0, 0.0

    max_dist = max(len_t, len_h) // 2
    match = 0
    penalizado_por_especificidad = False
    invalido_por_contradiccion = False
    
    hash_t = [0] * len_t
    hash_h = [0] * len_h

    for i in range(len_t):
        for j in range(max(0, i - max_dist), min(len_h, i + max_dist + 1)):
            if hash_h[j] == 0:
                # 1. Análisis de Polaridad (Negación)
                t_afirmativo = (i == 0 or T[i-1] not in palabras_negacion)
                h_afirmativo = (j == 0 or H[j-1] not in palabras_negacion)
                mismo_estado = (t_afirmativo == h_afirmativo)

                # 2. Evaluación de Relaciones Semánticas
                es_sin = (T[i] == H[j] or H[j] in sinT[i] or T[i] in sinH[j]
                          or sinT[i].intersection(sinH[j]))

                # Para RTE: H debe ser más general que T (Hiperónimo de T)
                es_gen = (H[j] in HipT[i] or T[i] in hipH[j]
                          or sinT[i].intersection(hipH[j]) 
                          or sinH[j].intersection(HipT[i]))
                # Para RTE: H debe ser más general que T (Hiperónimo de T)
                es_esp = (T[i] in HipH[j] or H[j] in hipT[i]
                          or sinT[i].intersection(HipH[j]) 
                          or sinH[j].intersection(hipT[i]))

                es_ant = (H[j] in antT[i] or T[i] in antH[j] 
                          or sinH[j].intersection(antT[i]) 
                          or sinT[i].intersection(antH[j]))

                # --- LÓGICA DE INFERENCIA (MATCHES) ---
                if mismo_estado:
                    if es_sin or es_gen:
                        match += 1
                        hash_t[i], hash_h[j] = 1, 1
                        break
                    elif es_esp:
                        # No sumamos match completo o marcamos para penalizar
                        penalizado_por_especificidad = True
                    elif es_ant:
                        # Afirmar un antónimo (T: "está vivo", H: "está muerto")
                        invalido_por_contradiccion = True
                else:
                    # Estados diferentes (uno negado, uno afirmativo)
                    if es_sin or es_gen or es_esp:
                        # T: "está vivo", H: "no está vivo" -> Contradicción
                        invalido_por_contradiccion = True
                    elif es_ant:
                        # T: "no está vivo", H: "está muerto" -> Entailment
                        match += 1
                        hash_t[i], hash_h[j] = 1, 1
                        break

    if match == 0: return 0.0, 0.0

    # 3. Transposiciones (con sensibilidad semántica coherente)
    t = 0
    p = 0
    for i in range(len_t):
        if hash_t[i]:
            # Buscamos la siguiente palabra en H que tuvo un match
            while p < len_h and hash_h[p] == 0: 
                p += 1
            
            if p < len_h:
                # --- LA CLAVE ESTÁ AQUÍ ---
                # Solo es transposición si NO hay ninguna relación de equivalencia o jerarquía válida
                es_equivalente = (T[i] == H[p] or 
                                H[p] in sinT[i] or 
                                T[i] in sinH[p] or 
                                sinT[i].intersection(sinH[p]))
                
                es_jerarquia_valida = (H[p] in HipT[i] or 
                                    T[i] in hipH[p] or 
                                    sinT[i].intersection(hipH[p]) or 
                                    sinH[p].intersection(HipT[i]))

                if not (es_equivalente or es_jerarquia_valida):
                    t += 1
                p += 1
    t /= 2.0

    # --- NUEVA SECCIÓN: PENALIZACIÓN POR INFORMACIÓN EXTRA (NEUTRAL) ---
    # Contamos cuántas palabras de la hipótesis no tuvieron pareja (hash_h[j] == 0)
    h_huerfanas = sum(1 for matched in hash_h if matched == 0)
    
    # Calculamos el ratio de "novedad" en H
    ratio_info_nueva = h_huerfanas / len_h if len_h > 0 else 0
    
    # --- FIN DE LA NUEVA SECCIÓN ---

    # 4. Score Final
    score_jaro = (match/len_t + match/len_h + (match-t)/match) / 3.0
    
    # Aplicar penalización "Kill-switch" si se detectó contradicción
    if invalido_por_contradiccion:
        score_jaro *= 0.01 
    elif penalizado_por_especificidad:
        score_jaro *= 0.2   # Penalización moderada (Falta de especificidad en T)

    # Si H tiene mucha información que T no mencionó, castigamos el score
    # para empujarlo hacia la zona de 'NEUTRAL' en tu gráfica.
    if ratio_info_nueva > 0.1: # Umbral ajustable (25% de info nueva)
        score_jaro *= (1.0 - (ratio_info_nueva * 0.4))
        
    return match / len_h, score_jaro

##########################################    TRANSFER ENTROPY      ############################
#informacon mutua y entrpi conjunta
def calculate_shannon_entropy(sequence):
    _, counts = np.unique(sequence, return_counts=True)
    return entropy(counts, base=2)

def calculate_joint_entropy(seq_p, seq_h):
    min_len = min(len(seq_p), len(seq_h))
    joint_seq = list(zip(seq_p[:min_len], seq_h[:min_len]))
    _, counts = np.unique(joint_seq, axis=0, return_counts=True)
    return entropy(counts, base=2)

def calculate_transfer_entropy(source, target, delay=1):
    min_len = min(len(source), len(target))
    source, target = source[:min_len], target[:min_len]
    N = len(target) - delay
    if N <= 0: return 0.0
    
    x_curr, x_past, y_past = target[delay:], target[:-delay], source[:-delay]
    c_3 = Counter(zip(x_curr, x_past, y_past))
    c_2_xp_yp = Counter(zip(x_past, y_past))
    c_2_xn_xp = Counter(zip(x_curr, x_past))
    c_1_xp = Counter(x_past)
    
    te = 0.0
    for (xn, xp, yp), count in c_3.items():
        p_joint = count / N
        p_cond_full = count / c_2_xp_yp[(xp, yp)]
        p_cond_limited = c_2_xn_xp[(xn, xp)] / c_1_xp[xp]
        if p_cond_full > 0 and p_cond_limited > 0:
            te += p_joint * np.log2(p_cond_full / p_cond_limited)
    return te

def calculos_mi_joint(p_signal, h_signal):
    h_h = calculate_shannon_entropy(h_signal)
    h_joint = calculate_joint_entropy(p_signal, h_signal)
    te = calculate_transfer_entropy(p_signal, h_signal)
    mi = h_p + h_h - h_joint

    # Clasificación
    ratio = te / h_h if h_h > 0 else 0
    if ratio > 0.25 and mi > 0.4:
        veredict, color = "ENTAILMENT", "green"
    elif mi < 0.2:
        veredict, color = "NEUTRAL", "blue"
    else:
        veredict, color = "CONTRADICTION / UNKNOWN", "red"
    
    return h_h,h_joint,te,mi,veredict
##########################################    CARGA  DE ARCHIVOS      ############################

#samples
prueba=pd.read_csv(raiz+"data/"+sys.argv[1])
#para obtener pesos
#prueba=pd.read_pickle("data/validacion/"+sys.argv[1]) # para obtener los pesos de las votaciones

textos = prueba["sentence1"].to_list()       # almacenamiento en listas
hipotesis = prueba["sentence2"].to_list()
clases = prueba["gold_label"].to_list()

# lista de listas para dataframes
new_data = {'Texto':[],'Hipotesis':[],'TextoL':[],'HipotesisL':[],'dicEntT':[],'dicEntH':[],
            'ConteosR':[],'ConteosG1':[],'ConteosG2':[],'ConteosG3':[],'ConteosG4':[],
            'Jaro-Winkler_rit':[],'Jaro-Winkler_rit1':[],'Jaro-Winkler_dsf':[],'Jaro-Winkler_dsf1':[],
            'Jaro-Winkler_rit2':[],'Jaro-Winkler_rit21':[],
            'Jaro-Winkler_rte_expert':[],'Jaro-Winkler_rte_expert1':[],'simBoWrel':[],'ranking':[],
            'Jaro-Winkler_ritR':[],'Jaro-Winkler_ritR1':[],
            'h_p':[],'h_h':[],'h_joint':[],'te':[],'mi':[],'veredict':[],'h_p_l':[],'h_h_l':[],'h_joint_l':[],'te_l':[],'mi_l':[],'veredict_l':[],
            'h_h_g1':[],'h_joint_g1':[],'te_g1':[],'mi_g1':[],'veredict_g1':[],
            'h_h_g2':[],'h_joint_g2':[],'te_g2':[],'mi_g2':[],'veredict_g2':[],
            'h_h_g3':[],'h_joint_g3':[],'te_g3':[],'mi_g3':[],'veredict_g3':[],
            'h_h_g4':[],'h_joint_g4':[],'te_g4':[],'mi_g4':[],'veredict_g4':[],
            'h_h_g1S':[],'h_joint_g1S':[],'te_g1S':[],'mi_g1S':[],'veredict_g1S':[],
            'h_h_g2S':[],'h_joint_g2S':[],'te_g2S':[],'mi_g2S':[],'veredict_g2S':[],
            'h_h_g3S':[],'h_joint_g3S':[],'te_g3S':[],'mi_g3S':[],'veredict_g3S':[],
            'h_h_g4S':[],'h_joint_g4S':[],'te_g4S':[],'mi_g4S':[],'veredict_g4S':[],
            'TH':[],'TH_G1':[],'TH_G2':[],'TH_G3':[],'TH_G4':[],'TH_G1_G2':[],'TH_G1_G3':[],'TH_G1_G4':[],'TH_G2_G4':[],'TH_G3_G4':[],'TH_l':[],'H_l':[],'I_l':[],'IF_l':[],'IM_l':[],
            'THS':[],'THS_G1':[],'THS_G2':[],'THS_G3':[],'THS_G4':[],'THS_G1_G2':[],'THS_G1_G3':[],'THS_G1_G4':[],'THS_G2_G4':[],'THS_G3_G4':[],
            'THC':[],'THC_G1':[],'THC_G2':[],'THC_G3':[],'THC_G4':[],'THC_G1_G2':[],'THC_G1_G3':[],'THC_G1_G4':[],'THC_G2_G4':[],'THC_G3_G4':[],
            'H':[],'H_G1':[],'H_G2':[],'H_G3':[],'H_G4':[],'H_G1_G2':[],'H_G1_G3':[],'H_G1_G4':[],'H_G2_G4':[],'H_G3_G4':[],'H_G2_G4':[],'H_G3_G4':[],
            'T_H':[],'T_H_G1':[],'T_H_G2':[],'T_H_G3':[],'T_H_G4':[],'T_H_G1_G2':[],'T_H_G1_G3':[],'T_H_G1_G4':[],'T_H_G2_G4':[],'T_H_G3_G4':[],
            'HS':[],'HS_G1':[],'HS_G2':[],'HS_G3':[],'HS_G4':[],'HS_G1_G2':[],'HS_G1_G3':[],'HS_G1_G4':[],'HS_G2_G4':[],'HS_G3_G4':[],
            'I':[],'I_G1':[],'I_G2':[],'I_G3':[],'I_G4':[],'I_G1_G2':[],'I_G1_G3':[],'I_G1_G4':[],'I_G2_G4':[],'I_G3_G4':[],
            'IF':[],'IF_G1':[],'IF_G2':[],'IF_G3':[],'IF_G4':[],'IF_G1_G2':[],'IF_G1_G3':[],'IF_G1_G4':[],'IF_G2_G4':[],'IF_G3_G4':[],
            'IM':[],'IM_G1':[],'IM_G2':[],'IM_G3':[],'IM_G4':[],'IM_G1_G2':[],'IM_G1_G3':[],'IM_G1_G4':[],'IM_G2_G4':[],'IM_G3_G4':[],
            'IMF':[],'IMF_G1':[],'IMF_G2':[],'IMF_G3':[],'IMF_G4':[],'IMF_G1_G2':[],'IMF_G1_G3':[],'IMF_G1_G4':[],'IMF_G2_G4':[],'IMF_G3_G4':[],    
            'xTH':[],'xTH_G1':[],'xTH_G2':[],'xTH_G3':[],'xTH_G4':[],'xTH_G1_G2':[],'xTH_G1_G3':[],'xTH_G1_G4':[],'xTH_G2_G4':[],'xTH_G3_G4':[],
            'xTHS':[],'xTHS_G1':[],'xTHS_G2':[],'xTHS_G3':[],'xTHS_G4':[],'xTHS_G1_G2':[],'xTHS_G1_G3':[],'xTHS_G1_G4':[],'xTHS_G2_G4':[],'xTHS_G3_G4':[],
            'xTHC':[],'xTHC_G1':[],'xTHC_G2':[],'xTHC_G3':[],'xTHC_G4':[],'xTHC_G1_G2':[],'xTHC_G1_G3':[],'xTHC_G1_G4':[],'xTHC_G2_G4':[],'xTHC_G3_G4':[],
            'xH':[],'xH_G1':[],'xH_G2':[],'xH_G3':[],'xH_G4':[],'xH_G1_G2':[],'xH_G1_G3':[],'xH_G1_G4':[],'xH_G2_G4':[],'xH_G3_G4':[],'xH_G2_G4':[],'xH_G3_G4':[],
            'xT_H':[],'xT_H_G1':[],'xT_H_G2':[],'xT_H_G3':[],'xT_H_G4':[],'xT_H_G1_G2':[],'xT_H_G1_G3':[],'xT_H_G1_G4':[],'xT_H_G2_G4':[],'xT_H_G3_G4':[],
            'xHS':[],'xHS_G1':[],'xHS_G2':[],'xHS_G3':[],'xHS_G4':[],'xHS_G1_G2':[],'xHS_G1_G3':[],'xHS_G1_G4':[],'xHS_G2_G4':[],'xHS_G3_G4':[],
            'xI':[],'xI_G1':[],'xI_G2':[],'xI_G3':[],'xI_G4':[],'xI_G1_G2':[],'xI_G1_G3':[],'xI_G1_G4':[],'xI_G2_G4':[],'xI_G3_G4':[],
            'xIF':[],'xIF_G1':[],'xIF_G2':[],'xIF_G3':[],'xIF_G4':[],'xIF_G1_G2':[],'xIF_G1_G3':[],'xIF_G1_G4':[],'xIF_G2_G4':[],'xIF_G3_G4':[],
            'xIM':[],'xIM_G1':[],'xIM_G2':[],'xIM_G3':[],'xIM_G4':[],'xIM_G1_G2':[],'xIM_G1_G3':[],'xIM_G1_G4':[],'xIM_G2_G4':[],'xIM_G3_G4':[],
            'xIMF':[],'xIMF_G1':[],'xIMF_G2':[],'xIMF_G3':[],'xIMF_G4':[],'xIMF_G1_G2':[],'xIMF_G1_G3':[],'xIMF_G1_G4':[],'xIMF_G2_G4':[],'xIMF_G3_G4':[],    
            'C':[],'C_G1':[],'C_G2':[],'C_G3':[],'C_G4':[],'C_G1_G2':[],'C_G1_G3':[],'C_G1_G4':[],'C_G2_G4':[],'C_G3_G4':[],
            's1':[],'s2':[],'sinT':[],'sinH':[],'antT':[],'antH':[],'HipT':[],'HipH':[],'hipT':[],'hipH':[],
            'clases' : []}

##########################################    INICIO DE PROCESO       ############################
inicio = time.time()
for i in range(len(textos)):
#for i in range(4):free s
    print(i) 
    texto_i=str(textos[i]).lower()
    hipotesis_i=str(hipotesis[i]).lower()
       
    #Revisar si es numerico la hipótesis para identificar en los resultados
    # para hacer el proceso o no
    if(type(hipotesis[i])==type(1.0) or type(textos[i])==type(1.0)):
        print("Falla")

        new_data['simBoWrel'].append(0)
        new_data['Jaro-Winkler_rit'].append(0)
        new_data['Jaro-Winkler_rit1'].append(0)
        new_data['Jaro-Winkler_rit2'].append(0)
        new_data['Jaro-Winkler_rit21'].append(0)
        new_data['Jaro-Winkler_ritR'].append(0)
        new_data['Jaro-Winkler_ritR1'].append(0)
        new_data['Jaro-Winkler_dsf'].append(0)
        new_data['Jaro-Winkler_dsf1'].append(0)
        new_data['Jaro-Winkler_rte_expert'].append(0)
        new_data['Jaro-Winkler_rte_expert1'].append(0)

        new_data['h_p_l'].append(0)
        new_data['h_h_l'].append(0)
        new_data['h_joint_l'].append(0)
        new_data['te_l'].append(0)
        new_data['mi_l'].append(0)
        new_data['veredict_l'].append(0)
        new_data['TH_l'].append(0)
        new_data['H_l'].append(0)
        new_data['I_l'].append(0)
        new_data['IF_l'].append(0)
        new_data['IM_l'].append(0)

        new_data['s1'].append(set())
        new_data['s2'].append(set())
        new_data['sinT'].append(set())
        new_data['sinH'].append(set())
        new_data['antT'].append(set())
        new_data['antH'].append(set())
        new_data['HipT'].append(set())
        new_data['HipH'].append(set())
        new_data['hipT'].append(set())
        new_data['hipH'].append(set())

        new_data['h_p'].append(0)
        new_data['h_h'].append(0)
        new_data['h_joint'].append(0)
        new_data['te'].append(0)
        new_data['mi'].append(0)
        new_data['veredict'].append(0)
        new_data['h_h_g1'].append(0)
        new_data['h_joint_g1'].append(0)
        new_data['te_g1'].append(0)
        new_data['mi_g1'].append(0)
        new_data['veredict_g1'].append(0)
        new_data['h_h_g2'].append(0)
        new_data['h_joint_g2'].append(0)
        new_data['te_g2'].append(0)
        new_data['mi_g2'].append(0)
        new_data['veredict_g2'].append(0)
        new_data['h_h_g3'].append(0)
        new_data['h_joint_g3'].append(0)
        new_data['te_g3'].append(0)
        new_data['mi_g3'].append(0)
        new_data['veredict_g3'].append(0)

        new_data['h_h_g4'].append(0)
        new_data['h_joint_g4'].append(0)
        new_data['te_g4'].append(0)
        new_data['mi_g4'].append(0)
        new_data['veredict_g4'].append(0)

        new_data['h_h_g1S'].append(0)
        new_data['h_joint_g1S'].append(0)
        new_data['te_g1S'].append(0)
        new_data['mi_g1S'].append(0)
        new_data['veredict_g1S'].append(0)
        new_data['h_h_g2S'].append(0)
        new_data['h_joint_g2S'].append(0)
        new_data['te_g2S'].append(0)
        new_data['mi_g2S'].append(0)
        new_data['veredict_g2S'].append(0)
        new_data['h_h_g3S'].append(0)
        new_data['h_joint_g3S'].append(0)
        new_data['te_g3S'].append(0)
        new_data['mi_g3S'].append(0)
        new_data['veredict_g3S'].append(0)
        new_data['h_h_g4S'].append(0)
        new_data['h_joint_g4S'].append(0)
        new_data['te_g4S'].append(0)
        new_data['mi_g4S'].append(0)
        new_data['veredict_g4S'].append(0)

        new_data['H'].append(0)
        new_data['H_G1'].append(0)
        new_data['H_G2'].append(0)
        new_data['H_G1_G2'].append(0)
        new_data['H_G3'].append(0)
        new_data['H_G1_G3'].append(0)
        new_data['H_G4'].append(0)
        new_data['H_G1_G4'].append(0)
        new_data['H_G2_G4'].append(0)
        new_data['H_G3_G4'].append(0)

        new_data['T_H'].append(0)
        new_data['T_H_G1'].append(0)
        new_data['T_H_G2'].append(0)
        new_data['T_H_G1_G2'].append(0)
        new_data['T_H_G3'].append(0)
        new_data['T_H_G1_G3'].append(0)
        new_data['T_H_G4'].append(0)
        new_data['T_H_G1_G4'].append(0)
        new_data['T_H_G2_G4'].append(0)
        new_data['T_H_G3_G4'].append(0)

        new_data['HS'].append(0)
        new_data['HS_G1'].append(0)
        new_data['HS_G2'].append(0)
        new_data['HS_G1_G2'].append(0)
        new_data['HS_G3'].append(0)
        new_data['HS_G1_G3'].append(0)
        new_data['HS_G4'].append(0)
        new_data['HS_G1_G4'].append(0)
        new_data['HS_G2_G4'].append(0)
        new_data['HS_G3_G4'].append(0)

        new_data['TH'].append(0)
        new_data['TH_G1'].append(0)
        new_data['TH_G2'].append(0)
        new_data['TH_G1_G2'].append(0)
        new_data['TH_G3'].append(0)
        new_data['TH_G1_G3'].append(0)
        new_data['TH_G4'].append(0)
        new_data['TH_G1_G4'].append(0)
        new_data['TH_G2_G4'].append(0)
        new_data['TH_G3_G4'].append(0)

        new_data['THS'].append(0)
        new_data['THS_G1'].append(0)
        new_data['THS_G2'].append(0)
        new_data['THS_G1_G2'].append(0)
        new_data['THS_G3'].append(0)
        new_data['THS_G1_G3'].append(0)
        new_data['THS_G4'].append(0)
        new_data['THS_G1_G4'].append(0)
        new_data['THS_G2_G4'].append(0)
        new_data['THS_G3_G4'].append(0)

        new_data['THC'].append(0)
        new_data['THC_G1'].append(0)
        new_data['THC_G2'].append(0)
        new_data['THC_G1_G2'].append(0)
        new_data['THC_G3'].append(0)
        new_data['THC_G1_G3'].append(0)
        new_data['THC_G4'].append(0)
        new_data['THC_G1_G4'].append(0)
        new_data['THC_G2_G4'].append(0)
        new_data['THC_G3_G4'].append(0)

        new_data['ranking'].append({})

        new_data['I'].append(0)
        new_data['I_G1'].append(0)
        new_data['I_G2'].append(0)
        new_data['I_G1_G2'].append(0)
        new_data['I_G3'].append(0)
        new_data['I_G1_G3'].append(0)
        new_data['I_G4'].append(0)
        new_data['I_G1_G4'].append(0)
        new_data['I_G2_G4'].append(0)
        new_data['I_G3_G4'].append(0)

        new_data['IF'].append(0)
        new_data['IF_G1'].append(0)
        new_data['IF_G2'].append(0)
        new_data['IF_G1_G2'].append(0)
        new_data['IF_G3'].append(0)
        new_data['IF_G1_G3'].append(0)
        new_data['IF_G4'].append(0)
        new_data['IF_G1_G4'].append(0)
        new_data['IF_G2_G4'].append(0)
        new_data['IF_G3_G4'].append(0)

        new_data['IM'].append(0)
        new_data['IM_G1'].append(0)
        new_data['IM_G2'].append(0)
        new_data['IM_G1_G2'].append(0)
        new_data['IM_G3'].append(0)
        new_data['IM_G1_G3'].append(0)
        new_data['IM_G4'].append(0)
        new_data['IM_G1_G4'].append(0)
        new_data['IM_G2_G4'].append(0)
        new_data['IM_G3_G4'].append(0)

        new_data['IMF'].append(0)
        new_data['IMF_G1'].append(0)
        new_data['IMF_G2'].append(0)
        new_data['IMF_G1_G2'].append(0)
        new_data['IMF_G3'].append(0)
        new_data['IMF_G1_G3'].append(0)
        new_data['IMF_G4'].append(0)
        new_data['IMF_G1_G4'].append(0)
        new_data['IMF_G2_G4'].append(0)
        new_data['IMF_G3_G4'].append(0)

        new_data['xH'].append(0)
        new_data['xH_G1'].append(0)
        new_data['xH_G2'].append(0)
        new_data['xH_G1_G2'].append(0)
        new_data['xH_G3'].append(0)
        new_data['xH_G1_G3'].append(0)
        new_data['xH_G4'].append(0)
        new_data['xH_G1_G4'].append(0)
        new_data['xH_G2_G4'].append(0)
        new_data['xH_G3_G4'].append(0)

        new_data['xT_H'].append(0)
        new_data['xT_H_G1'].append(0)
        new_data['xT_H_G2'].append(0)
        new_data['xT_H_G1_G2'].append(0)
        new_data['xT_H_G3'].append(0)
        new_data['xT_H_G1_G3'].append(0)
        new_data['xT_H_G4'].append(0)
        new_data['xT_H_G1_G4'].append(0)
        new_data['xT_H_G2_G4'].append(0)
        new_data['xT_H_G3_G4'].append(0)

        new_data['xHS'].append(0)
        new_data['xHS_G1'].append(0)
        new_data['xHS_G2'].append(0)
        new_data['xHS_G1_G2'].append(0)
        new_data['xHS_G3'].append(0)
        new_data['xHS_G1_G3'].append(0)
        new_data['xHS_G4'].append(0)
        new_data['xHS_G1_G4'].append(0)
        new_data['xHS_G2_G4'].append(0)
        new_data['xHS_G3_G4'].append(0)

        new_data['xTH'].append(0)
        new_data['xTH_G1'].append(0)
        new_data['xTH_G2'].append(0)
        new_data['xTH_G1_G2'].append(0)
        new_data['xTH_G3'].append(0)
        new_data['xTH_G1_G3'].append(0)
        new_data['xTH_G4'].append(0)
        new_data['xTH_G1_G4'].append(0)
        new_data['xTH_G2_G4'].append(0)
        new_data['xTH_G3_G4'].append(0)

        new_data['xTHS'].append(0)
        new_data['xTHS_G1'].append(0)
        new_data['xTHS_G2'].append(0)
        new_data['xTHS_G1_G2'].append(0)
        new_data['xTHS_G3'].append(0)
        new_data['xTHS_G1_G3'].append(0)
        new_data['xTHS_G4'].append(0)
        new_data['xTHS_G1_G4'].append(0)
        new_data['xTHS_G2_G4'].append(0)
        new_data['xTHS_G3_G4'].append(0)

        new_data['xTHC'].append(0)
        new_data['xTHC_G1'].append(0)
        new_data['xTHC_G2'].append(0)
        new_data['xTHC_G1_G2'].append(0)
        new_data['xTHC_G3'].append(0)
        new_data['xTHC_G1_G3'].append(0)
        new_data['xTHC_G4'].append(0)
        new_data['xTHC_G1_G4'].append(0)
        new_data['xTHC_G2_G4'].append(0)
        new_data['xTHC_G3_G4'].append(0)

        new_data['xI'].append(0)
        new_data['xI_G1'].append(0)
        new_data['xI_G2'].append(0)
        new_data['xI_G1_G2'].append(0)
        new_data['xI_G3'].append(0)
        new_data['xI_G1_G3'].append(0)
        new_data['xI_G4'].append(0)
        new_data['xI_G1_G4'].append(0)
        new_data['xI_G2_G4'].append(0)
        new_data['xI_G3_G4'].append(0)

        new_data['xIF'].append(0)
        new_data['xIF_G1'].append(0)
        new_data['xIF_G2'].append(0)
        new_data['xIF_G1_G2'].append(0)
        new_data['xIF_G3'].append(0)
        new_data['xIF_G1_G3'].append(0)
        new_data['xIF_G4'].append(0)
        new_data['xIF_G1_G4'].append(0)
        new_data['xIF_G2_G4'].append(0)
        new_data['xIF_G3_G4'].append(0)

        new_data['xIM'].append(0)
        new_data['xIM_G1'].append(0)
        new_data['xIM_G2'].append(0)
        new_data['xIM_G1_G2'].append(0)
        new_data['xIM_G3'].append(0)
        new_data['xIM_G1_G3'].append(0)
        new_data['xIM_G4'].append(0)
        new_data['xIM_G1_G4'].append(0)
        new_data['xIM_G2_G4'].append(0)
        new_data['xIM_G3_G4'].append(0)

        new_data['xIMF'].append(0)
        new_data['xIMF_G1'].append(0)
        new_data['xIMF_G2'].append(0)
        new_data['xIMF_G1_G2'].append(0)
        new_data['xIMF_G3'].append(0)
        new_data['xIMF_G1_G3'].append(0)
        new_data['xIMF_G4'].append(0)
        new_data['xIMF_G1_G4'].append(0)
        new_data['xIMF_G2_G4'].append(0)
        new_data['xIMF_G3_G4'].append(0)

        new_data['C'].append(0)
        new_data['C_G1'].append(0)
        new_data['C_G2'].append(0)
        new_data['C_G1_G2'].append(0)
        new_data['C_G3'].append(0)          
        new_data['C_G1_G3'].append(0)
        new_data['C_G4'].append(0)          
        new_data['C_G1_G4'].append(0)
        new_data['C_G2_G4'].append(0)
        new_data['C_G3_G4'].append(0)
        
        new_data['Texto'].append(texto_i)
        new_data['Hipotesis'].append(hipotesis_i)
        new_data['TextoL'].append([])
        new_data['HipotesisL'].append([])
        new_data['dicEntT'].append([])
        new_data['dicEntH'].append([])
        new_data['ConteosR'].append([])
        new_data['ConteosG1'].append([])
        new_data['ConteosG2'].append([])
        new_data['ConteosG3'].append([])
        new_data['ConteosG4'].append([])
        new_data['clases'].append(9)
    else:
        print("Correcto")
        print(texto_i)
        r_t,t_clean_m,lemmas_t,pos_t=representacion_entidadesDavidSetM(nlp,texto_i.translate(str.maketrans('', '', string.punctuation)))
        print(r_t)

        print(hipotesis_i)
        r_h,h_clean_m,lemmas_h,pos_h=representacion_entidadesDavidSetM(nlp,hipotesis_i.translate(str.maketrans('', '', string.punctuation)))
        print(r_h)

        # lista de relaciones
        lista_rel_G1=[]
        lista_rel_G2=[]
        lista_rel_G3=[]
        lista_rel_G4=[]

        #palabras de cada grupo
        borrar_g1=set()
        borrar_g2=set()
        borrar_g3=set()
        borrar_g4=set()

        #palabras de cada grupo
        t_borrar_g1=set()
        t_borrar_g2=set()
        t_borrar_g3=set()
        t_borrar_g4=set()

        # primero evaluar si existen acronimos que se puedan identificar
        sinT=[]
        sinH=[]
        for t in lemmas_h:
            if t in df_diccionario_generales:
                sinH.append((df_diccionario_generales[t]["synonym"]).union(df_diccionario_especificas[t]["synonym"]))
        print(sinH)
        new_text=" ".join(lemmas_t) # reconstuir el texto
        words_found=[]
        print("--------------------------------------------------------")
        for e_i in range(len(sinH)):
            for e_syn in sinH[e_i]:
                if "_" in e_syn:
                    nsin=str(e_syn).replace("_"," ")
                    #print(nsin)
                    if(" "+nsin+" " in new_text):
                        #lista_rel_G1.append((lemmas_h[e_i],"synonym",nsin))
                        #words_found.append(lemmas_h[e_i])
                        for n_s in nsin.split():
                            print("se encontro primero",lemmas_h[e_i],n_s)
                            if n_s in r_t and lemmas_h[e_i] in r_h:
                                verif_att,att_t,att_h,cat=check_atributos(r_t[n_s],r_h[lemmas_h[e_i]])
                                if verif_att:
                                    lista_rel_G1.append((att_t+" "+n_s,"is_a",att_h+" "+lemmas_h[e_i]))
                                    words_found.append(lemmas_h[e_i])
                                    borrar_g1.add(lemmas_h[e_i])
                                    t_borrar_g1.add(n_s)
                                    matching=True
                                    break
                                else:
                                    if cat==2:
                                        lista_rel_G3.append((att_h+" "+lemmas_h[e_i],"is_a",att_t+" "+n_s))
                                        words_found.append(lemmas_h[e_i])
                                        borrar_g3.add(lemmas_h[e_i])
                                        t_borrar_g3.add(n_s)
                                        matching=True
                                        break
                                    else:
                                        lista_rel_G2.append((att_t+" "+n_s,"distinct_from",att_h+" "+lemmas_h[e_i])) ######### ESTE ES EL PROBLEMA
                                        words_found.append(lemmas_h[e_i])
                                        borrar_g2.add(lemmas_h[e_i])
                                        t_borrar_g2.add(n_s)
                                        matching=True
                                        break
        print(new_text)
        print(words_found)

        # Jaro-Winckler
        s1=lemmas_t[:]
        s2=lemmas_h[:]
        # obtener todos los sinonimos
        sinT=[]
        antT=[]
        HipT=[]
        hipT=[]

        sinH=[]
        antH=[]
        HipH=[]
        hipH=[]

        #relT=[]
        #relH=[]
        
        # # encontrar bolsa de sinonimos de cada token
        for t in s1:
            if t in df_diccionario_generales:
                sinT.append((df_diccionario_generales[t]["synonym"]).union(df_diccionario_especificas[t]["synonym"]))
                antT.append((df_diccionario_generales[t]["antonym"]).union(df_diccionario_generales[t]["distinct_from"]))
                #relT.append((df_diccionario_generales[t]["related_to"]).union(df_diccionario_generales[t]["similar_to"]))
                temp_set=set()
                for p_g in relaciones_gen:
                    temp_set=temp_set.union(df_diccionario_generales[t][p_g])
                HipT.append(temp_set)
                temp_set=set()
                for p_g in relaciones_gen:
                    temp_set=temp_set.union(df_diccionario_especificas[t][p_g])
                hipT.append(temp_set)
            else:
                sinT.append(set(t))
                HipT.append(set(t))
                hipT.append(set(t))
                antT.append(set(t))
                #relT.append(set(t))
        for h in s2:
            if h in df_diccionario_generales:
                sinH.append((df_diccionario_generales[h]["synonym"]).union(df_diccionario_especificas[h]["synonym"]))
                antH.append((df_diccionario_generales[h]["antonym"]).union(df_diccionario_generales[h]["distinct_from"]))
                #relH.append((df_diccionario_generales[h]["related_to"]).union(df_diccionario_generales[h]["similar_to"]))
                temp_set=set()
                # se usan las mismas relacines pero en el otro sentido para la hiponimia
                for p_g in relaciones_gen:
                    temp_set=temp_set.union(df_diccionario_especificas[h][p_g])
                hipH.append(temp_set)
                temp_set=set()
                for p_g in relaciones_gen:
                    temp_set=temp_set.union(df_diccionario_generales[h][p_g])
                HipH.append(temp_set)
            else:
                sinH.append(set(h))
                hipH.append(set(h)) 
                HipH.append(set(h)) 
                antH.append(set(h))
                #relH.append(set(h))
                
        print(sinT)
        print(sinH)
        
        #jaro_sim,jaro_winkler=jaro_distance(s1, s2,sinT,sinH,HipT,hipH)
        jaro_sim,jaro_winkler=jaro_torres_rte_expert(s1, s2, sinT, sinH, antT, antH, HipT, HipH, hipT, hipH, palabras_negacion_adicionales)
        new_data['Jaro-Winkler_rit'].append(jaro_sim)
        new_data['Jaro-Winkler_rit1'].append(jaro_winkler)

        jaro_sim,jaro_winkler=jaro_rte_expert(s2, s1, sinH, sinT, antH, antT, HipH, HipT, palabras_negacion_adicionales)
        new_data['Jaro-Winkler_rit2'].append(jaro_sim)
        new_data['Jaro-Winkler_rit21'].append(jaro_winkler)

        jaro_sim,jaro_winkler=jaro_torres_rte_expert(s2, s1, sinH, sinT, antH, antT, HipH, HipT, hipH, hipT, palabras_negacion_adicionales)
        new_data['Jaro-Winkler_ritR'].append(jaro_sim)
        new_data['Jaro-Winkler_ritR1'].append(jaro_winkler)

        new_data['s1'].append(s1)
        new_data['s2'].append(s2)
        new_data['sinT'].append(sinT)
        new_data['sinH'].append(sinH)
        new_data['antT'].append(antT)
        new_data['antH'].append(antH)
        new_data['HipT'].append(HipT)
        new_data['HipH'].append(HipH)
        new_data['hipT'].append(hipT)
        new_data['hipH'].append(hipH)

        #jaro_sim,jaro_winkler=jaro_distance_semantic(s1, s2,sinT,sinH,HipT,hipH)
        jaro_sim,jaro_winkler=jaro_rte_expert(s1, s2, sinT, sinH, hipT, hipH, HipT, hipH, palabras_negacion_adicionales)
        new_data['Jaro-Winkler_dsf'].append(jaro_sim)
        new_data['Jaro-Winkler_dsf1'].append(jaro_winkler)

        jaro_sim,jaro_winkler=jaro_rte_expert(s1, s2, sinT, sinH, antT, antH, HipT, hipH, palabras_negacion_adicionales)
        new_data['Jaro-Winkler_rte_expert'].append(jaro_sim)
        new_data['Jaro-Winkler_rte_expert1'].append(jaro_winkler)

        # BoW mejorado con sinonimos, hiperonimos
        print(s1)
        print(s2)
        for nh in range(len(s2)):
            for nt in range(len(sinT)):
                if(s2[nh] in sinT[nt] or s2[nh] in HipT[nt]):
                    print("Revision de reemplazo",s2[nh],s1[nt],sinT[nt],HipT[nt])
                    if(s2[nh]!=s1[nt]):
                        print(s2[nh],s1[nt])
                        s2[nh]=s1[nt]
                elif(s1[nt] in sinH[nh] or s1[nt] in hipH[nh]):
                    print("Revision de reemplazo",s1[nt],s2[nh],sinH[nh],hipH[nh])
                    if(s1[nt]!=s2[nh]):
                        print(s1[nt],s2[nh])
                        s1[nt]=s2[nh]

        print("Después de la revisión")
        print(s1)
        print(s2)
        # crear las bolsas de palabras

        v_vocabulario=set(s1).union(set(s2))
        t_bow=[]
        h_bow=[]
        for e in v_vocabulario:
            t_bow.append(s1.count(e))
            h_bow.append(s2.count(e))
        
        print(t_bow)
        print(h_bow)
        
        resultBoW = 1 - spatial.distance.cosine(t_bow, h_bow)
        print("BoW: ",resultBoW)
        new_data['simBoWrel'].append(resultBoW)

        # Matriz de alineamiento para probar la contención de las entidades

        t_vectors_n=get_matrix_rep2(t_clean_m, nlp, normed=True)
        h_vectors_n=get_matrix_rep2(h_clean_m, nlp, normed=True)

        redondeo=2
        ma_n=np.dot(t_vectors_n,h_vectors_n.T)
        ma_n = np.clip(ma_n, 0, 1).round(redondeo)
        ma=pd.DataFrame(ma_n,index=t_clean_m,columns=h_clean_m)
        print(ma)

        

        top_k=3
        # # #PARA REVISAR SI EXISTEN RELACIONES DE SIMILITUD SEMÁNTICA A TRAVÉS DEL USO DE CONCEPNET
        print("proceso lexico")
        print(ma,ma.columns)

        for c_c in ma.columns:
            if c_c not in words_found:
                print("columna a checar",c_c)
                # filtrar el top 3 de los mejores similitud coseno para cada token de H vs tokens de T que sean mayores a 0
                # una vez que encontremos quien se sale del ciclo
                if ma.columns.tolist().count(c_c)>1:
                    temp=ma.loc[:, c_c].iloc[:, 0].sort_values(ascending=False)
                else:
                    temp=ma[c_c].sort_values(ascending=False)
                ranks=list(temp[:top_k].index)
                valranks=list(temp[:top_k].values)
                #print(valranks,ranks)
                matching=False
                for r_i in range(len(ranks)): 
                    print("busqeuda",r_i,c_c,ranks[r_i])
                    # relaciones generales
                    verificacion,rel_found=check_entail_syn(ranks[r_i],c_c) #COMPARACION SINONIMOS DE wt y wh
                    if verificacion:
                        print("sindasdada",ranks[r_i],c_c,rel_found)
                        verif_att,att_t,att_h,cat=check_atributos(r_t[ranks[r_i]],r_h[c_c])
                        if verif_att:
                            lista_rel_G1.append((att_t+" "+ranks[r_i],rel_found,att_h+" "+c_c))
                            borrar_g1.add(c_c)
                            t_borrar_g1.add(ranks[r_i])
                            matching=True
                            break
                        else:
                            if cat==2:
                                lista_rel_G3.append((att_h+" "+c_c,"is_a",att_t+" "+ranks[r_i]))
                                borrar_g3.add(c_c)
                                t_borrar_g3.add(ranks[r_i])
                                matching=True
                                break
                            else:
                                lista_rel_G2.append((att_t+" "+ranks[r_i],"distinct_from",att_h+" "+c_c)) ######### ESTE ES EL PROBLEMA
                                borrar_g2.add(c_c)
                                t_borrar_g2.add(ranks[r_i])
                                matching=True
                                break
                    # checo en relaciones generales si se encuentra el token de la hipótesis
                    verificacion,rel_found=check_entail_gen(ranks[r_i],c_c) #COMPARACION GENERAL DE wt y wh
                    if verificacion:
                        print("gensadas",ranks[r_i],c_c,rel_found)
                        verif_att,att_t,att_h,cat=check_atributos(r_t[ranks[r_i]],r_h[c_c])
                        if verif_att:
                            lista_rel_G1.append((att_t+" "+ranks[r_i],rel_found,att_h+" "+c_c))
                            borrar_g1.add(c_c)
                            t_borrar_g1.add(ranks[r_i])
                            matching=True
                            break
                        else:
                            if cat==2:
                                lista_rel_G3.append((att_h+" "+c_c,"is_a",att_t+" "+ranks[r_i]))
                                borrar_g3.add(c_c)
                                t_borrar_g3.add(ranks[r_i])
                                matching=True
                                break
                            else:
                                lista_rel_G2.append((att_t+" "+ranks[r_i],"distinct_from",att_h+" "+c_c))
                                borrar_g2.add(c_c)
                                t_borrar_g2.add(ranks[r_i])
                                matching=True
                                break
                    # contradicciones
                    verificacion,rel_found=check_contradiction_ant(ranks[r_i],c_c) #COMPARACION CONTRA DE wt y wh
                    if verificacion:
                        att_t,att_h=get_atributos(r_t[ranks[r_i]],r_h[c_c])
                        lista_rel_G2.append((att_t+" "+ranks[r_i],rel_found,att_h+" "+c_c))
                        borrar_g2.add(c_c)
                        t_borrar_g2.add(ranks[r_i])
                        matching=True
                        break
                    #conjuntos de información adicional part_of escalar y subir
                    # verificacion,rel_found=check_info_adicional(ranks[r_i],c_c)#COMPARACION NEUTRAL DE wt y wh
                    # if verificacion:
                    #     verif_att,att_t,att_h,cat=check_atributos(r_t[ranks[r_i]],r_h[c_c])
                    #     if verif_att:
                    #         lista_rel_G1.append((att_t+" "+ranks[r_i],rel_found,att_h+" "+c_c))
                    #         borrar_g1.add(c_c)
                    #         t_borrar_g1.add(ranks[r_i])
                    #         matching=True
                    #         break
                    #     else:
                    #         if cat==2:
                    #             lista_rel_G3.append((att_h+" "+c_c,"is_a",att_t+" "+ranks[r_i]))
                    #             borrar_g3.add(c_c)
                    #             t_borrar_g3.add(ranks[r_i])
                    #             matching=True
                    #             break
                    #         else:
                    #             lista_rel_G2.append((att_t+" "+ranks[r_i],"distinct_from",att_h+" "+c_c))
                    #             borrar_g2.add(c_c)
                    #             t_borrar_g2.add(ranks[r_i])
                    #             matching=True
                    #             break
                    #especificas
                    #verificacion,rel_found=check_neutral_rel(ranks[r_i],c_c)
                    verificacion,rel_found=check_entail_gen(c_c,ranks[r_i])
                    if verificacion:
                        lista_rel_G3.append((c_c,rel_found,ranks[r_i]))
                        borrar_g3.add(c_c)
                        t_borrar_g3.add(ranks[r_i])
                        matching=True
                        break
                if matching==False:
                    lista_rel_G4.append(("","unknown",c_c))
                    borrar_g4.add(c_c)
                    # if valranks[r_i]>0.9:
                    #     lista_rel_G1.append(("","similitud",c_c))
                    #     borrar_g1.add(c_c)
                    # elif valranks[r_i]>=0.2 and valranks[r_i]<=0.9:
                    #     lista_rel_G3.append(("","rel",c_c))
                    #     borrar_g3.add(c_c)
                    # if valranks[r_i]<0.1:
                    #     lista_rel_G2.append(("","no_rel",c_c))
                    #     borrar_g2.add(c_c)
                    # else:
                    #     lista_rel_G4.append(("","unknown",c_c))
                    #     borrar_g4.add(c_c)

        lista_rel_ST=[]
        lista_rel_ST.extend(lista_rel_G1)
        lista_rel_ST.extend(lista_rel_G2)
        lista_rel_ST.extend(lista_rel_G3)
        lista_rel_ST.extend(lista_rel_G4)
        print("Relaciones encontradas G 1:",borrar_g1)
        print("Relaciones encontradas G 2:",borrar_g2)
        print("Relaciones encontradas G 3:",borrar_g3)

         
        #################################################################################################

        #Matriz sin relaciones encontradas
        t_df_g1 = ma.drop(index=list(t_borrar_g1),columns=list(borrar_g1))
        t_df_g2 = ma.drop(index=list(t_borrar_g2),columns=list(borrar_g2))
        t_df_g1_g2 = ma.drop(index=list(t_borrar_g1)+list(t_borrar_g2),columns=list(borrar_g1)+list(borrar_g2))
        t_df_g3 = ma.drop(index=list(t_borrar_g3),columns=list(borrar_g3))
        t_df_g1_g3 = ma.drop(index=list(t_borrar_g1)+list(t_borrar_g3),columns=list(borrar_g1)+list(borrar_g3))
        t_df_g4 = ma.drop(index=list(t_borrar_g4),columns=list(borrar_g4))
        t_df_g1_g4 = ma.drop(index=list(t_borrar_g1)+list(t_borrar_g4),columns=list(borrar_g1)+list(borrar_g4))
        t_df_g3_g4 = ma.drop(index=list(t_borrar_g3)+list(t_borrar_g4),columns=list(borrar_g3)+list(borrar_g4))
        t_df_g2_g4 = ma.drop(index=list(t_borrar_g2)+list(t_borrar_g4),columns=list(borrar_g2)+list(borrar_g4))

###################33

        #Matriz sin relaciones encontradas
        df_g1 = ma.drop(columns=list(borrar_g1))
        df_g2 = ma.drop(columns=list(borrar_g2))
        df_g1_g2 = ma.drop(columns=list(borrar_g1)+list(borrar_g2))
        df_g3 = ma.drop(columns=list(borrar_g3))
        df_g1_g3 = ma.drop(columns=list(borrar_g1)+list(borrar_g3))
        df_g4 = ma.drop(columns=list(borrar_g4))
        df_g1_g4 = ma.drop(columns=list(borrar_g1)+list(borrar_g4))
        df_g3_g4 = ma.drop(columns=list(borrar_g3)+list(borrar_g4))
        df_g2_g4 = ma.drop(columns=list(borrar_g2)+list(borrar_g4))

        #Matriz de solo relaciones encontradas
        dfS_g1 = ma[list(borrar_g1)]
        dfS_g2 = ma[list(borrar_g2)]
        dfS_g1_g2 = ma[list(borrar_g1)+list(borrar_g2)]
        dfS_g3 = ma[list(borrar_g3)]
        dfS_g1_g3 = ma[list(borrar_g1)+list(borrar_g3)]
        dfS_g4 = ma[list(borrar_g4)]
        dfS_g1_g4 = ma[list(borrar_g1)+list(borrar_g4)]
        dfS_g2_g3_g4 = ma[list(borrar_g2)+list(borrar_g3)+list(borrar_g4)]
        dfS_g3_g4 = ma[list(borrar_g3)+list(borrar_g4)]
        dfS_g2_g4 = ma[list(borrar_g2)+list(borrar_g4)]

        # mutual information y entropia conjunta
        p_signal = ma.values.flatten()
        h_signal = ma.T.values.flatten()

        # Cálculos
        h_p = calculate_shannon_entropy(p_signal)
        h_h,h_joint,te,mi,veredict = calculos_mi_joint(p_signal,h_signal)

        new_data['h_p'].append(h_p)
        new_data['h_h'].append(h_h)
        new_data['h_joint'].append(h_joint)
        new_data['te'].append(te)
        new_data['mi'].append(mi)
        new_data['veredict'].append(veredict)

        h_h,h_joint,te,mi,veredict = calculos_mi_joint(p_signal,df_g1.T.values.flatten())
        new_data['h_h_g1'].append(h_h)
        new_data['h_joint_g1'].append(h_joint)
        new_data['te_g1'].append(te)
        new_data['mi_g1'].append(mi)
        new_data['veredict_g1'].append(veredict)

        h_h,h_joint,te,mi,veredict = calculos_mi_joint(p_signal,df_g2.T.values.flatten())
        new_data['h_h_g2'].append(h_h)
        new_data['h_joint_g2'].append(h_joint)
        new_data['te_g2'].append(te)
        new_data['mi_g2'].append(mi)
        new_data['veredict_g2'].append(veredict)

        h_h,h_joint,te,mi,veredict = calculos_mi_joint(p_signal,df_g3.T.values.flatten())
        new_data['h_h_g3'].append(h_h)
        new_data['h_joint_g3'].append(h_joint)
        new_data['te_g3'].append(te)
        new_data['mi_g3'].append(mi)
        new_data['veredict_g3'].append(veredict)

        h_h,h_joint,te,mi,veredict = calculos_mi_joint(p_signal,df_g4.T.values.flatten())
        new_data['h_h_g4'].append(h_h)
        new_data['h_joint_g4'].append(h_joint)
        new_data['te_g4'].append(te)
        new_data['mi_g4'].append(mi)
        new_data['veredict_g4'].append(veredict)

        h_h,h_joint,te,mi,veredict = calculos_mi_joint(p_signal,dfS_g1.T.values.flatten())
        new_data['h_h_g1S'].append(h_h)
        new_data['h_joint_g1S'].append(h_joint)
        new_data['te_g1S'].append(te)
        new_data['mi_g1S'].append(mi)
        new_data['veredict_g1S'].append(veredict)

        h_h,h_joint,te,mi,veredict = calculos_mi_joint(p_signal,dfS_g2.T.values.flatten())
        new_data['h_h_g2S'].append(h_h)
        new_data['h_joint_g2S'].append(h_joint)
        new_data['te_g2S'].append(te)
        new_data['mi_g2S'].append(mi)
        new_data['veredict_g2S'].append(veredict)

        h_h,h_joint,te,mi,veredict = calculos_mi_joint(p_signal,dfS_g3.T.values.flatten())
        new_data['h_h_g3S'].append(h_h)
        new_data['h_joint_g3S'].append(h_joint)
        new_data['te_g3S'].append(te)
        new_data['mi_g3S'].append(mi)
        new_data['veredict_g3S'].append(veredict)

        h_h,h_joint,te,mi,veredict = calculos_mi_joint(p_signal,dfS_g4.T.values.flatten())
        new_data['h_h_g4S'].append(h_h)
        new_data['h_joint_g4S'].append(h_joint)
        new_data['te_g4S'].append(te)
        new_data['mi_g4S'].append(mi)
        new_data['veredict_g4S'].append(veredict)

        # Transfer entropy de la matriz sin relaciones        
        new_data['TH'].append(calculate_te_safe(ma.round(1).values.flatten(),ma.round(1).T.values.flatten()))
        new_data['TH_G1'].append(calculate_te_safe(df_g1.round(1).values.flatten(),df_g1.round(1).T.values.flatten()))
        new_data['TH_G2'].append(calculate_te_safe(df_g2.round(1).values.flatten(),df_g2.round(1).T.values.flatten()))
        new_data['TH_G1_G2'].append(calculate_te_safe(df_g1_g2.round(1).values.flatten(),df_g1_g2.round(1).T.values.flatten()))
        new_data['TH_G3'].append(calculate_te_safe(df_g3.round(1).values.flatten(),df_g3.round(1).T.values.flatten()))
        new_data['TH_G1_G3'].append(calculate_te_safe(df_g1_g3.round(1).values.flatten(),df_g1_g3.round(1).T.values.flatten()))
        new_data['TH_G4'].append(calculate_te_safe(df_g4.round(1).values.flatten(),df_g4.round(1).T.values.flatten()))
        new_data['TH_G1_G4'].append(calculate_te_safe(df_g1_g4.round(1).values.flatten(),df_g1_g4.round(1).T.values.flatten()))
        new_data['TH_G2_G4'].append(calculate_te_safe(df_g2_g4.round(1).values.flatten(),df_g2_g4.round(1).T.values.flatten()))
        new_data['TH_G3_G4'].append(calculate_te_safe(df_g3_g4.round(1).values.flatten(),df_g3_g4.round(1).T.values.flatten()))

        # Transfer entropy de T a la matriz sin relaciones
        new_data['THS'].append(calculate_te_safe(ma.round(1).values.flatten(),ma.round(1).T.values.flatten()))
        new_data['THS_G1'].append(calculate_te_safe(ma.round(1).values.flatten(),df_g1.round(1).T.values.flatten()))
        new_data['THS_G2'].append(calculate_te_safe(ma.round(1).values.flatten(),df_g2.round(1).T.values.flatten()))
        new_data['THS_G1_G2'].append(calculate_te_safe(ma.round(1).values.flatten(),df_g1_g2.round(1).T.values.flatten()))
        new_data['THS_G3'].append(calculate_te_safe(ma.round(1).values.flatten(),df_g3.round(1).T.values.flatten()))
        new_data['THS_G1_G3'].append(calculate_te_safe(ma.round(1).values.flatten(),df_g1_g3.round(1).T.values.flatten()))
        new_data['THS_G4'].append(calculate_te_safe(ma.round(1).values.flatten(),df_g4.round(1).T.values.flatten()))
        new_data['THS_G1_G4'].append(calculate_te_safe(ma.round(1).values.flatten(),df_g1_g4.round(1).T.values.flatten()))
        new_data['THS_G2_G4'].append(calculate_te_safe(ma.round(1).values.flatten(),df_g2_g4.round(1).T.values.flatten()))
        new_data['THS_G3_G4'].append(calculate_te_safe(ma.round(1).values.flatten(),df_g3_g4.round(1).T.values.flatten()))

        # Transfer entropy de T a la matriz con relaciones de 
        new_data['THC'].append(calculate_te_safe(ma.round(1).values.flatten(),ma.round(1).T.values.flatten()))
        new_data['THC_G1'].append(calculate_te_safe(ma.round(1).values.flatten(),dfS_g1.round(1).T.values.flatten()))
        new_data['THC_G2'].append(calculate_te_safe(ma.round(1).values.flatten(),dfS_g2.round(1).T.values.flatten()))
        new_data['THC_G1_G2'].append(calculate_te_safe(ma.round(1).values.flatten(),dfS_g1_g2.round(1).T.values.flatten()))
        new_data['THC_G3'].append(calculate_te_safe(ma.round(1).values.flatten(),dfS_g3.round(1).T.values.flatten()))
        new_data['THC_G1_G3'].append(calculate_te_safe(ma.round(1).values.flatten(),dfS_g1_g3.round(1).T.values.flatten()))
        new_data['THC_G4'].append(calculate_te_safe(ma.round(1).values.flatten(),dfS_g4.round(1).T.values.flatten()))
        new_data['THC_G1_G4'].append(calculate_te_safe(ma.round(1).values.flatten(),dfS_g1_g4.round(1).T.values.flatten()))
        new_data['THC_G2_G4'].append(calculate_te_safe(ma.round(1).values.flatten(),dfS_g2_g4.round(1).T.values.flatten()))
        new_data['THC_G3_G4'].append(calculate_te_safe(ma.round(1).values.flatten(),dfS_g3_g4.round(1).T.values.flatten()))

        try:
            new_data['ranking'].append(find_key_informant_words(ma.round(1)))
        except:
            new_data['ranking'].append({})

        #Entropia de valores de similitud de la matriz sin relaciones
        new_data['T_H'].append(entropia(ma.round(1).values.flatten()))
        new_data['T_H_G1'].append(entropia(t_df_g1.round(1).values.flatten()))
        new_data['T_H_G2'].append(entropia(t_df_g2.round(1).values.flatten()))
        new_data['T_H_G1_G2'].append(entropia(t_df_g1_g2.round(1).values.flatten()))
        new_data['T_H_G3'].append(entropia(t_df_g3.round(1).values.flatten()))
        new_data['T_H_G1_G3'].append(entropia(t_df_g1_g3.round(1).values.flatten()))
        new_data['T_H_G4'].append(entropia(t_df_g4.round(1).values.flatten()))
        new_data['T_H_G1_G4'].append(entropia(t_df_g1_g4.round(1).values.flatten()))
        new_data['T_H_G2_G4'].append(entropia(t_df_g2_g4.round(1).values.flatten()))
        new_data['T_H_G3_G4'].append(entropia(t_df_g3_g4.round(1).values.flatten()))

        #Entropia de valores de similitud de la matriz sin relaciones
        new_data['H'].append(entropia(ma.round(1).values.flatten()))
        new_data['H_G1'].append(entropia(df_g1.round(1).values.flatten()))
        new_data['H_G2'].append(entropia(df_g2.round(1).values.flatten()))
        new_data['H_G1_G2'].append(entropia(df_g1_g2.round(1).values.flatten()))
        new_data['H_G3'].append(entropia(df_g3.round(1).values.flatten()))
        new_data['H_G1_G3'].append(entropia(df_g1_g3.round(1).values.flatten()))
        new_data['H_G4'].append(entropia(df_g4.round(1).values.flatten()))
        new_data['H_G1_G4'].append(entropia(df_g1_g4.round(1).values.flatten()))
        new_data['H_G2_G4'].append(entropia(df_g2_g4.round(1).values.flatten()))
        new_data['H_G3_G4'].append(entropia(df_g3_g4.round(1).values.flatten()))

        #Entropia de valores de similitud de la matriz de relaciones 
        new_data['HS'].append(entropia(dfS_g2_g3_g4.round(1).values.flatten()))
        new_data['HS_G1'].append(entropia(dfS_g1.round(1).values.flatten()))
        new_data['HS_G2'].append(entropia(dfS_g2.round(1).values.flatten()))
        new_data['HS_G1_G2'].append(entropia(dfS_g1_g2.round(1).values.flatten()))
        new_data['HS_G3'].append(entropia(dfS_g3.round(1).values.flatten()))
        new_data['HS_G1_G3'].append(entropia(dfS_g1_g3.round(1).values.flatten()))
        new_data['HS_G4'].append(entropia(dfS_g4.round(1).values.flatten()))
        new_data['HS_G1_G4'].append(entropia(dfS_g1_g4.round(1).values.flatten()))
        new_data['HS_G2_G4'].append(entropia(dfS_g2_g4.round(1).values.flatten()))
        new_data['HS_G3_G4'].append(entropia(dfS_g3_g4.round(1).values.flatten()))

        #suma de valores de similitud de la matriz quitando relaciones dividida entre el número de elementos
        new_data['I'].append(ma.sum().sum()/(ma.shape[1]*ma.shape[0]))
        new_data['I_G1'].append(df_g1.sum().sum()/(df_g1.shape[1]*df_g1.shape[0]))
        new_data['I_G2'].append(df_g2.sum().sum()/(df_g2.shape[1]*df_g2.shape[0]))
        new_data['I_G1_G2'].append(df_g1_g2.sum().sum()/(df_g1_g2.shape[1]*df_g1_g2.shape[0]))
        new_data['I_G3'].append(df_g3.sum().sum()/(df_g3.shape[1]*df_g3.shape[0]))
        new_data['I_G1_G3'].append(df_g1_g3.sum().sum()/(df_g1_g3.shape[1]*df_g1_g3.shape[0]))
        new_data['I_G4'].append(df_g4.sum().sum()/(df_g4.shape[1]*df_g4.shape[0]))
        new_data['I_G1_G4'].append(df_g1_g4.sum().sum()/(df_g1_g4.shape[1]*df_g1_g4.shape[0]))
        new_data['I_G2_G4'].append(df_g2_g4.sum().sum()/(df_g2_g4.shape[1]*df_g2_g4.shape[0]))
        new_data['I_G3_G4'].append(df_g3_g4.sum().sum()/(df_g3_g4.shape[1]*df_g3_g4.shape[0]))

        #suma de valores de similitud de la matriz de relaciones dividida entre el número de elementos
        new_data['IF'].append(ma.sum().sum()/(ma.shape[1]*ma.shape[0]))
        new_data['IF_G1'].append(dfS_g1.sum().sum()/(dfS_g1.shape[1]*dfS_g1.shape[0]))
        new_data['IF_G2'].append(dfS_g2.sum().sum()/(dfS_g2.shape[1]*dfS_g2.shape[0]))
        new_data['IF_G1_G2'].append(dfS_g1_g2.sum().sum()/(dfS_g1_g2.shape[1]*dfS_g1_g2.shape[0]))
        new_data['IF_G3'].append(dfS_g3.sum().sum()/(dfS_g3.shape[1]*dfS_g3.shape[0]))
        new_data['IF_G1_G3'].append(dfS_g1_g3.sum().sum()/(dfS_g1_g3.shape[1]*dfS_g1_g3.shape[0]))
        new_data['IF_G4'].append(dfS_g4.sum().sum()/(dfS_g4.shape[1]*dfS_g4.shape[0]))
        new_data['IF_G1_G4'].append(dfS_g1_g4.sum().sum()/(dfS_g1_g4.shape[1]*dfS_g1_g4.shape[0]))
        new_data['IF_G2_G4'].append(dfS_g2_g4.sum().sum()/(dfS_g2_g4.shape[1]*dfS_g2_g4.shape[0]))
        new_data['IF_G3_G4'].append(dfS_g3_g4.sum().sum()/(dfS_g3_g4.shape[1]*dfS_g3_g4.shape[0]))

        #suma de valores maximos de similitud de la matriz quitando relaciones dividida entre el número de elementos
        new_data['IM'].append(ma.max().sum()/(ma.shape[1]))
        new_data['IM_G1'].append(df_g1.max().sum()/(df_g1.shape[1]))
        new_data['IM_G2'].append(df_g2.max().sum()/(df_g2.shape[1]))
        new_data['IM_G1_G2'].append(df_g1_g2.max().sum()/(df_g1_g2.shape[1]))
        new_data['IM_G3'].append(df_g3.max().sum()/(df_g3.shape[1]))
        new_data['IM_G1_G3'].append(df_g1_g3.max().sum()/(df_g1_g3.shape[1]))
        new_data['IM_G4'].append(df_g4.max().sum()/(df_g4.shape[1]))
        new_data['IM_G1_G4'].append(df_g1_g4.max().sum()/(df_g1_g4.shape[1]))
        new_data['IM_G2_G4'].append(df_g2_g4.max().sum()/(df_g2_g4.shape[1]))
        new_data['IM_G3_G4'].append(df_g3_g4.max().sum()/(df_g3_g4.shape[1]))

        #suma de valores maximosde similitud de la matriz de relaciones dividida entre el número de elementos
        new_data['IMF'].append(dfS_g2_g3_g4.max().sum()/(ma.shape[1]))
        new_data['IMF_G1'].append(dfS_g1.max().sum()/(dfS_g1.shape[1]))
        new_data['IMF_G2'].append(dfS_g2.max().sum()/(dfS_g2.shape[1]))
        new_data['IMF_G1_G2'].append(dfS_g1_g2.max().sum()/(dfS_g1_g2.shape[1]))
        new_data['IMF_G3'].append(dfS_g3.max().sum()/(dfS_g3.shape[1]))
        new_data['IMF_G1_G3'].append(dfS_g1_g3.max().sum()/(dfS_g1_g3.shape[1]))
        new_data['IMF_G4'].append(dfS_g4.max().sum()/(dfS_g4.shape[1]))
        new_data['IMF_G1_G4'].append(dfS_g1_g4.max().sum()/(dfS_g1_g4.shape[1]))
        new_data['IMF_G2_G4'].append(dfS_g2_g4.max().sum()/(dfS_g2_g4.shape[1]))
        new_data['IMF_G3_G4'].append(dfS_g3_g4.max().sum()/(dfS_g3_g4.shape[1]))

        # Proporcion de conteo de relaciones encontradas dividida entre total de relaciones
        new_data['C'].append(len(lista_rel_ST[:]))
        new_data['C_G1'].append(len(lista_rel_G1[:])/len(lista_rel_ST[:]))
        new_data['C_G2'].append(len(lista_rel_G2[:])/len(lista_rel_ST[:]))
        new_data['C_G1_G2'].append((len(lista_rel_G1[:])+len(lista_rel_G2[:]))/len(lista_rel_ST[:]))
        new_data['C_G3'].append(len(lista_rel_G3[:])/len(lista_rel_ST[:]))          
        new_data['C_G1_G3'].append((len(lista_rel_G1[:])+len(lista_rel_G3[:]))/len(lista_rel_ST[:]))
        new_data['C_G4'].append(len(lista_rel_G4[:])/len(lista_rel_ST[:]))          
        new_data['C_G1_G4'].append((len(lista_rel_G1[:])+len(lista_rel_G4[:]))/len(lista_rel_ST[:]))
        new_data['C_G2_G4'].append((len(lista_rel_G2[:])+len(lista_rel_G4[:]))/len(lista_rel_ST[:]))
        new_data['C_G3_G4'].append((len(lista_rel_G3[:])+len(lista_rel_G4[:]))/len(lista_rel_ST[:]))


        # identificar las entidades de g4 a nivel lexico
        l_borrar_g1=set()
        l_borrar_g2=set()
        l_borrar_g3=set()
        l_borrar_g4=set()
        for e in borrar_g1:
            l_borrar_g1.add(e)
            l_borrar_g1=l_borrar_g1.union(r_h[e])
        for e in borrar_g2:
            l_borrar_g2.add(e)
            l_borrar_g2=l_borrar_g2.union(r_h[e])
        for e in borrar_g3:
            l_borrar_g3.add(e)
            l_borrar_g3=l_borrar_g3.union(r_h[e])
        for e in borrar_g4:
            l_borrar_g4.add(e)
            l_borrar_g4=l_borrar_g4.union(r_h[e])

        # Matriz lexica
        t_vectors_n_l=get_matrix_rep2(lemmas_t, nlp, normed=True)
        h_vectors_n_l=get_matrix_rep2(lemmas_h, nlp, normed=True)

        redondeo=2
        ma_n_l=np.dot(t_vectors_n_l,h_vectors_n_l.T)
        ma_n_l = np.clip(ma_n_l, 0, 1).round(redondeo)
        ma=pd.DataFrame(ma_n_l,index=lemmas_t,columns=lemmas_h)
        print(ma)

        # mutual information y entropia conjunta
        p_signal = ma.values.flatten()
        h_signal = ma.T.values.flatten()

        # Cálculos
        h_p = calculate_shannon_entropy(p_signal)
        h_h,h_joint,te,mi,veredict = calculos_mi_joint(p_signal,h_signal)

        new_data['h_p_l'].append(h_p)
        new_data['h_h_l'].append(h_h)
        new_data['h_joint_l'].append(h_joint)
        new_data['te_l'].append(te)
        new_data['mi_l'].append(mi)
        new_data['veredict_l'].append(veredict)
        new_data['TH_l'].append(calculate_te_safe(ma.round(1).values.flatten(),ma.round(1).T.values.flatten()))
        new_data['H_l'].append(entropia(ma.round(1).values.flatten()))
        new_data['I_l'].append(ma.sum().sum()/(ma.shape[1]*ma.shape[0]))
        new_data['IF_l'].append(ma.sum().sum()/(ma.shape[1]*ma.shape[0]))
        new_data['IM_l'].append(ma.max().sum()/(ma.shape[1]))


              #Matriz sin relaciones encontradas
        df_g1 = ma.drop(columns=list(l_borrar_g1))
        df_g2 = ma.drop(columns=list(l_borrar_g2))
        df_g1_g2 = ma.drop(columns=list(l_borrar_g1)+list(l_borrar_g2))
        df_g3 = ma.drop(columns=list(l_borrar_g3))
        df_g1_g3 = ma.drop(columns=list(l_borrar_g1)+list(l_borrar_g3))
        df_g4 = ma.drop(columns=list(l_borrar_g4))
        df_g1_g4 = ma.drop(columns=list(l_borrar_g1)+list(l_borrar_g4))
        df_g3_g4 = ma.drop(columns=list(l_borrar_g3)+list(l_borrar_g4))
        df_g2_g4 = ma.drop(columns=list(l_borrar_g2)+list(l_borrar_g4))

        #Matriz de solo relaciones encontradas
        dfS_g1 = ma[list(l_borrar_g1)]
        dfS_g2 = ma[list(l_borrar_g2)]
        dfS_g1_g2 = ma[list(l_borrar_g1)+list(l_borrar_g2)]
        dfS_g3 = ma[list(l_borrar_g3)]
        dfS_g1_g3 = ma[list(l_borrar_g1)+list(l_borrar_g3)]
        dfS_g4 = ma[list(l_borrar_g4)]
        dfS_g1_g4 = ma[list(l_borrar_g1)+list(l_borrar_g4)]
        dfS_g2_g3_g4 = ma[list(l_borrar_g2)+list(l_borrar_g3)+list(l_borrar_g4)]
        dfS_g3_g4 = ma[list(l_borrar_g3)+list(l_borrar_g4)]
        dfS_g2_g4 = ma[list(l_borrar_g2)+list(l_borrar_g4)]

        #Entropia de valores de similitud de la matriz sin relaciones
        new_data['xT_H'].append(entropia(ma.round(1).values.flatten()))
        new_data['xT_H_G1'].append(entropia(t_df_g1.round(1).values.flatten()))
        new_data['xT_H_G2'].append(entropia(t_df_g2.round(1).values.flatten()))
        new_data['xT_H_G1_G2'].append(entropia(t_df_g1_g2.round(1).values.flatten()))
        new_data['xT_H_G3'].append(entropia(t_df_g3.round(1).values.flatten()))
        new_data['xT_H_G1_G3'].append(entropia(t_df_g1_g3.round(1).values.flatten()))
        new_data['xT_H_G4'].append(entropia(t_df_g4.round(1).values.flatten()))
        new_data['xT_H_G1_G4'].append(entropia(t_df_g1_g4.round(1).values.flatten()))
        new_data['xT_H_G2_G4'].append(entropia(t_df_g2_g4.round(1).values.flatten()))
        new_data['xT_H_G3_G4'].append(entropia(t_df_g3_g4.round(1).values.flatten()))

        # Transfer entropy de la matriz sin relaciones        
        new_data['xTH'].append(calculate_te_safe(ma.round(1).values.flatten(),ma.round(1).T.values.flatten()))
        new_data['xTH_G1'].append(calculate_te_safe(df_g1.round(1).values.flatten(),df_g1.round(1).T.values.flatten()))
        new_data['xTH_G2'].append(calculate_te_safe(df_g2.round(1).values.flatten(),df_g2.round(1).T.values.flatten()))
        new_data['xTH_G1_G2'].append(calculate_te_safe(df_g1_g2.round(1).values.flatten(),df_g1_g2.round(1).T.values.flatten()))
        new_data['xTH_G3'].append(calculate_te_safe(df_g3.round(1).values.flatten(),df_g3.round(1).T.values.flatten()))
        new_data['xTH_G1_G3'].append(calculate_te_safe(df_g1_g3.round(1).values.flatten(),df_g1_g3.round(1).T.values.flatten()))
        new_data['xTH_G4'].append(calculate_te_safe(df_g4.round(1).values.flatten(),df_g4.round(1).T.values.flatten()))
        new_data['xTH_G1_G4'].append(calculate_te_safe(df_g1_g4.round(1).values.flatten(),df_g1_g4.round(1).T.values.flatten()))
        new_data['xTH_G2_G4'].append(calculate_te_safe(df_g2_g4.round(1).values.flatten(),df_g2_g4.round(1).T.values.flatten()))
        new_data['xTH_G3_G4'].append(calculate_te_safe(df_g3_g4.round(1).values.flatten(),df_g3_g4.round(1).T.values.flatten()))

        # Transfer entropy de T a la matriz sin relaciones
        new_data['xTHS'].append(calculate_te_safe(ma.round(1).values.flatten(),ma.round(1).T.values.flatten()))
        new_data['xTHS_G1'].append(calculate_te_safe(ma.round(1).values.flatten(),df_g1.round(1).T.values.flatten()))
        new_data['xTHS_G2'].append(calculate_te_safe(ma.round(1).values.flatten(),df_g2.round(1).T.values.flatten()))
        new_data['xTHS_G1_G2'].append(calculate_te_safe(ma.round(1).values.flatten(),df_g1_g2.round(1).T.values.flatten()))
        new_data['xTHS_G3'].append(calculate_te_safe(ma.round(1).values.flatten(),df_g3.round(1).T.values.flatten()))
        new_data['xTHS_G1_G3'].append(calculate_te_safe(ma.round(1).values.flatten(),df_g1_g3.round(1).T.values.flatten()))
        new_data['xTHS_G4'].append(calculate_te_safe(ma.round(1).values.flatten(),df_g4.round(1).T.values.flatten()))
        new_data['xTHS_G1_G4'].append(calculate_te_safe(ma.round(1).values.flatten(),df_g1_g4.round(1).T.values.flatten()))
        new_data['xTHS_G2_G4'].append(calculate_te_safe(ma.round(1).values.flatten(),df_g2_g4.round(1).T.values.flatten()))
        new_data['xTHS_G3_G4'].append(calculate_te_safe(ma.round(1).values.flatten(),df_g3_g4.round(1).T.values.flatten()))

        # Transfer entropy de T a la matriz con relaciones de 
        new_data['xTHC'].append(calculate_te_safe(ma.round(1).values.flatten(),ma.round(1).T.values.flatten()))
        new_data['xTHC_G1'].append(calculate_te_safe(ma.round(1).values.flatten(),dfS_g1.round(1).T.values.flatten()))
        new_data['xTHC_G2'].append(calculate_te_safe(ma.round(1).values.flatten(),dfS_g2.round(1).T.values.flatten()))
        new_data['xTHC_G1_G2'].append(calculate_te_safe(ma.round(1).values.flatten(),dfS_g1_g2.round(1).T.values.flatten()))
        new_data['xTHC_G3'].append(calculate_te_safe(ma.round(1).values.flatten(),dfS_g3.round(1).T.values.flatten()))
        new_data['xTHC_G1_G3'].append(calculate_te_safe(ma.round(1).values.flatten(),dfS_g1_g3.round(1).T.values.flatten()))
        new_data['xTHC_G4'].append(calculate_te_safe(ma.round(1).values.flatten(),dfS_g4.round(1).T.values.flatten()))
        new_data['xTHC_G1_G4'].append(calculate_te_safe(ma.round(1).values.flatten(),dfS_g1_g4.round(1).T.values.flatten()))
        new_data['xTHC_G2_G4'].append(calculate_te_safe(ma.round(1).values.flatten(),dfS_g2_g4.round(1).T.values.flatten()))
        new_data['xTHC_G3_G4'].append(calculate_te_safe(ma.round(1).values.flatten(),dfS_g3_g4.round(1).T.values.flatten()))

        #Entropia de valores de similitud de la matriz sin relaciones
        new_data['xH'].append(entropia(ma.round(1).values.flatten()))
        new_data['xH_G1'].append(entropia(df_g1.round(1).values.flatten()))
        new_data['xH_G2'].append(entropia(df_g2.round(1).values.flatten()))
        new_data['xH_G1_G2'].append(entropia(df_g1_g2.round(1).values.flatten()))
        new_data['xH_G3'].append(entropia(df_g3.round(1).values.flatten()))
        new_data['xH_G1_G3'].append(entropia(df_g1_g3.round(1).values.flatten()))
        new_data['xH_G4'].append(entropia(df_g4.round(1).values.flatten()))
        new_data['xH_G1_G4'].append(entropia(df_g1_g4.round(1).values.flatten()))
        new_data['xH_G2_G4'].append(entropia(df_g2_g4.round(1).values.flatten()))
        new_data['xH_G3_G4'].append(entropia(df_g3_g4.round(1).values.flatten()))

        #Entropia de valores de similitud de la matriz de relaciones 
        new_data['xHS'].append(entropia(dfS_g2_g3_g4.round(1).values.flatten()))
        new_data['xHS_G1'].append(entropia(dfS_g1.round(1).values.flatten()))
        new_data['xHS_G2'].append(entropia(dfS_g2.round(1).values.flatten()))
        new_data['xHS_G1_G2'].append(entropia(dfS_g1_g2.round(1).values.flatten()))
        new_data['xHS_G3'].append(entropia(dfS_g3.round(1).values.flatten()))
        new_data['xHS_G1_G3'].append(entropia(dfS_g1_g3.round(1).values.flatten()))
        new_data['xHS_G4'].append(entropia(dfS_g4.round(1).values.flatten()))
        new_data['xHS_G1_G4'].append(entropia(dfS_g1_g4.round(1).values.flatten()))
        new_data['xHS_G2_G4'].append(entropia(dfS_g2_g4.round(1).values.flatten()))
        new_data['xHS_G3_G4'].append(entropia(dfS_g3_g4.round(1).values.flatten()))

        #suma de valores de similitud de la matriz quitando relaciones dividida entre el número de elementos
        new_data['xI'].append(ma.sum().sum()/(ma.shape[1]*ma.shape[0]))
        new_data['xI_G1'].append(df_g1.sum().sum()/(df_g1.shape[1]*df_g1.shape[0]))
        new_data['xI_G2'].append(df_g2.sum().sum()/(df_g2.shape[1]*df_g2.shape[0]))
        new_data['xI_G1_G2'].append(df_g1_g2.sum().sum()/(df_g1_g2.shape[1]*df_g1_g2.shape[0]))
        new_data['xI_G3'].append(df_g3.sum().sum()/(df_g3.shape[1]*df_g3.shape[0]))
        new_data['xI_G1_G3'].append(df_g1_g3.sum().sum()/(df_g1_g3.shape[1]*df_g1_g3.shape[0]))
        new_data['xI_G4'].append(df_g4.sum().sum()/(df_g4.shape[1]*df_g4.shape[0]))
        new_data['xI_G1_G4'].append(df_g1_g4.sum().sum()/(df_g1_g4.shape[1]*df_g1_g4.shape[0]))
        new_data['xI_G2_G4'].append(df_g2_g4.sum().sum()/(df_g2_g4.shape[1]*df_g2_g4.shape[0]))
        new_data['xI_G3_G4'].append(df_g3_g4.sum().sum()/(df_g3_g4.shape[1]*df_g3_g4.shape[0]))

        #suma de valores de similitud de la matriz de relaciones dividida entre el número de elementos
        new_data['xIF'].append(ma.sum().sum()/(ma.shape[1]*ma.shape[0]))
        new_data['xIF_G1'].append(dfS_g1.sum().sum()/(dfS_g1.shape[1]*dfS_g1.shape[0]))
        new_data['xIF_G2'].append(dfS_g2.sum().sum()/(dfS_g2.shape[1]*dfS_g2.shape[0]))
        new_data['xIF_G1_G2'].append(dfS_g1_g2.sum().sum()/(dfS_g1_g2.shape[1]*dfS_g1_g2.shape[0]))
        new_data['xIF_G3'].append(dfS_g3.sum().sum()/(dfS_g3.shape[1]*dfS_g3.shape[0]))
        new_data['xIF_G1_G3'].append(dfS_g1_g3.sum().sum()/(dfS_g1_g3.shape[1]*dfS_g1_g3.shape[0]))
        new_data['xIF_G4'].append(dfS_g4.sum().sum()/(dfS_g4.shape[1]*dfS_g4.shape[0]))
        new_data['xIF_G1_G4'].append(dfS_g1_g4.sum().sum()/(dfS_g1_g4.shape[1]*dfS_g1_g4.shape[0]))
        new_data['xIF_G2_G4'].append(dfS_g2_g4.sum().sum()/(dfS_g2_g4.shape[1]*dfS_g2_g4.shape[0]))
        new_data['xIF_G3_G4'].append(dfS_g3_g4.sum().sum()/(dfS_g3_g4.shape[1]*dfS_g3_g4.shape[0]))

        #suma de valores maximos de similitud de la matriz quitando relaciones dividida entre el número de elementos
        new_data['xIM'].append(ma.max().sum()/(ma.shape[1]))
        new_data['xIM_G1'].append(df_g1.max().sum()/(df_g1.shape[1]))
        new_data['xIM_G2'].append(df_g2.max().sum()/(df_g2.shape[1]))
        new_data['xIM_G1_G2'].append(df_g1_g2.max().sum()/(df_g1_g2.shape[1]))
        new_data['xIM_G3'].append(df_g3.max().sum()/(df_g3.shape[1]))
        new_data['xIM_G1_G3'].append(df_g1_g3.max().sum()/(df_g1_g3.shape[1]))
        new_data['xIM_G4'].append(df_g4.max().sum()/(df_g4.shape[1]))
        new_data['xIM_G1_G4'].append(df_g1_g4.max().sum()/(df_g1_g4.shape[1]))
        new_data['xIM_G2_G4'].append(df_g2_g4.max().sum()/(df_g2_g4.shape[1]))
        new_data['xIM_G3_G4'].append(df_g3_g4.max().sum()/(df_g3_g4.shape[1]))

        #suma de valores maximosde similitud de la matriz de relaciones dividida entre el número de elementos
        new_data['xIMF'].append(dfS_g2_g3_g4.max().sum()/(ma.shape[1]))
        new_data['xIMF_G1'].append(dfS_g1.max().sum()/(dfS_g1.shape[1]))
        new_data['xIMF_G2'].append(dfS_g2.max().sum()/(dfS_g2.shape[1]))
        new_data['xIMF_G1_G2'].append(dfS_g1_g2.max().sum()/(dfS_g1_g2.shape[1]))
        new_data['xIMF_G3'].append(dfS_g3.max().sum()/(dfS_g3.shape[1]))
        new_data['xIMF_G1_G3'].append(dfS_g1_g3.max().sum()/(dfS_g1_g3.shape[1]))
        new_data['xIMF_G4'].append(dfS_g4.max().sum()/(dfS_g4.shape[1]))
        new_data['xIMF_G1_G4'].append(dfS_g1_g4.max().sum()/(dfS_g1_g4.shape[1]))
        new_data['xIMF_G2_G4'].append(dfS_g2_g4.max().sum()/(dfS_g2_g4.shape[1]))
        new_data['xIMF_G3_G4'].append(dfS_g3_g4.max().sum()/(dfS_g3_g4.shape[1]))


        new_data['Texto'].append(texto_i)
        new_data['Hipotesis'].append(hipotesis_i)
        new_data['TextoL'].append(lemmas_t)
        new_data['HipotesisL'].append(lemmas_h)
        new_data['dicEntT'].append(r_t)
        new_data['dicEntH'].append(r_h)
        new_data['ConteosR'].append(lista_rel_ST[:])
        new_data['ConteosG1'].append(lista_rel_G1[:])
        new_data['ConteosG2'].append(lista_rel_G2[:])
        new_data['ConteosG3'].append(lista_rel_G3[:])
        new_data['ConteosG4'].append(lista_rel_G4[:])
        new_data['clases'].append(clases[i])

df_resultados = pd.DataFrame(new_data)
#salida de muestreos
df_resultados.to_pickle("output/relationships/"+sys.argv[1]+".pickle") #cambiar a solo numero para rapido procesamiento
#salida de pesos
#df_resultados.to_pickle("salida/validacion/"+sys.argv[1]+"_.pickle") #cambiar a solo numero para rapido procesamiento
fin = time.time()
print("Tiempo que se llevo:",round(fin-inicio,2)," segundos")