import pandas as pd

products = pd.read_csv("olist_products_dataset.csv")

category_map = {}

for candidate in ["product_category_name_translation.csv",
                   "olist_product_category_name_translation.csv"]:
    try:
        trans = pd.read_csv(candidate)
        category_map = dict(zip(trans["product_category_name"],
                                 trans["product_category_name_english"]))
        print(f"Loaded {len(category_map)} official category translations from {candidate}")
        break
    except FileNotFoundError:
        continue

if not category_map:
    print("WARNING: official translation file not found — relying entirely on word-level fallback.")

# Word-level fallback dictionary for categories the official file misses
PT_WORD_MAP = {
    "agro": "agriculture", "industria": "industry", "comercio": "commerce",
    "moveis": "furniture", "decoracao": "decor", "beleza": "beauty", "saude": "health",
    "esporte": "sports", "lazer": "leisure", "informatica": "computing",
    "acessorios": "accessories", "cama": "bed", "mesa": "table", "banho": "bath",
    "eletronicos": "electronics", "eletrodomesticos": "appliances", "eletroportateis": "small appliances",
    "brinquedos": "toys", "livros": "books", "papelaria": "stationery",
    "alimentos": "food", "bebidas": "beverages", "construcao": "construction",
    "ferramentas": "tools", "automotivo": "automotive", "telefonia": "telephony",
    "climatizacao": "climate control", "cine": "cinema", "foto": "photo", "audio": "audio",
    "relogios": "watches", "presentes": "gifts", "festas": "party", "artigos": "goods",
    "utilidades": "utilities", "domesticas": "household", "fashion": "fashion",
    "roupa": "clothing", "roupas": "clothing", "calcados": "footwear", "infantil": "children's",
    "bebe": "baby", "malas": "bags", "seguros": "insurance", "servicos": "services",
    "portateis": "portable", "cozinha": "kitchen", "preparadores": "preparation",
    "sinalizacao": "signage", "musicais": "musical", "instrumentos": "instruments",
    "pequenos": "small", "grandes": "large", "cool": "novelty", "pet": "pet",
    "consoles": "consoles", "jogos": "games", "papel": "paper", "linha": "line",
    "cuidado": "care", "pcs": "PCs", "de": "of", "e": "and", "para": "for", "na": "in",
}

def word_level_fallback(name):
    words = str(name).split("_")
    translated = [PT_WORD_MAP.get(w.lower(), w) for w in words]
    return " ".join(translated).title()

normalized_category_map = {str(k).strip().lower(): v for k, v in category_map.items()}

CATEGORY_CORRECTIONS = {
    "Fashio Female Clothing": "Fashion Female Clothing",
    "Fashio Male Clothing": "Fashion Male Clothing",
    "Fashio Sport": "Fashion Sport",
    "Fashio Underwear Beach": "Fashion Underwear Beach",
    "Fashio Childrens Clothes": "Fashion Childrens Clothing",
    "Fashio Shoes": "Fashion Shoes",
    "Fashio Bags Accessories": "Fashion Bags Accessories",
    "Portateis Cozinha E Preparadores De Alimentos": "Portable Kitchen Appliances And Food Preparation",
}

def translate_category(raw_val):
    key = str(raw_val).strip().lower()
    if key in normalized_category_map:
        result = normalized_category_map[key].replace("_", " ").title()
    else:
        result = word_level_fallback(raw_val)
    return CATEGORY_CORRECTIONS.get(result, result)

unmatched_categories = sorted(set(
    val for val in products["product_category_name"].dropna().unique()
    if str(val).strip().lower() not in normalized_category_map
))
if unmatched_categories:
    print(f"{len(unmatched_categories)} categories not in official file — using word-level fallback for:")
    for c in unmatched_categories:
        print(f"  {c}  ->  {translate_category(c)}")

products["product_category_name"] = products["product_category_name"].apply(
    lambda x: translate_category(x) if pd.notna(x) else x
)
products.to_csv("olist_products_dataset_en.csv", index=False)
print("Saved: olist_products_dataset_en.csv\n")

payments = pd.read_csv("olist_order_payments_dataset.csv")

payment_type_map = {
    "credit_card": "Credit Card",
    "boleto": "Bank Transfer",
    "voucher": "Voucher",
    "debit_card": "Debit Card",
    "not_defined": "Other",
}
payments["payment_type"] = payments["payment_type"].map(payment_type_map).fillna(
    payments["payment_type"].astype(str).str.replace("_", " ").str.title()
)
payments.to_csv("olist_order_payments_dataset_en.csv", index=False)
print("Saved: olist_order_payments_dataset_en.csv\n")

#Customer-Seller State: Brazilian State codes converted to state names for ease of understanding

brazil_state_names = {
    "SP": "São Paulo", "MG": "Minas Gerais", "RJ": "Rio de Janeiro", "BA": "Bahia",
    "PR": "Paraná", "RS": "Rio Grande do Sul", "PE": "Pernambuco", "CE": "Ceará",
    "PA": "Pará", "SC": "Santa Catarina", "MA": "Maranhão", "GO": "Goiás",
    "AM": "Amazonas", "ES": "Espírito Santo", "PB": "Paraíba", "RN": "Rio Grande do Norte",
    "MT": "Mato Grosso", "AL": "Alagoas", "PI": "Piauí", "DF": "Distrito Federal",
    "MS": "Mato Grosso do Sul", "SE": "Sergipe", "RO": "Rondônia", "TO": "Tocantins",
    "AC": "Acre", "AP": "Amapá", "RR": "Roraima",
}

customers = pd.read_csv("olist_customers_dataset.csv")
customers["customer_state"] = customers["customer_state"].map(brazil_to_india_state).fillna(
    customers["customer_state"]
)
customers.to_csv("olist_customers_dataset_en.csv", index=False)
print("Saved: olist_customers_dataset_en.csv")

try:
    sellers = pd.read_csv("olist_sellers_dataset.csv")
    sellers["seller_state"] = sellers["seller_state"].map(brazil_to_india_state).fillna(
        sellers["seller_state"]
    )
    sellers.to_csv("olist_sellers_dataset_en.csv", index=False)
    print("Saved: olist_sellers_dataset_en.csv")
except FileNotFoundError:
    print("olist_sellers_dataset.csv not found — skipped (not required unless you use seller_state).")

print("\nDone. Update main.py to read the _en files instead of the raw ones (see next message).")
