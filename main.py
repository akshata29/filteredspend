import streamlit as st
import pandas as pd
import uuid
import json
import random
from io import StringIO
from azure.cosmos import CosmosClient, PartitionKey
from azure.identity import DefaultAzureCredential
from azure.cosmos.partition_key import PartitionKey
from azure.identity import DefaultAzureCredential
from openai import OpenAI, AzureOpenAI, AsyncAzureOpenAI


# --------------- Configuration ---------------
COSMOS_ENDPOINT = st.secrets.get("COSMOSDB_ENDPOINT")
COSMOS_DATABASE = st.secrets.get("COSMOSDB_DATABASE", "filteredspend")
COSMOS_CATALOG_CONTAINER = st.secrets.get("COSMOS_CATALOG_CONTAINER", "ProductCatalog")
COSMOS_APL_CONTAINER = st.secrets.get("COSMOS_APL_CONTAINER", "AllowedProductList")

OPENAI_ENDPOINT = st.secrets.get("AZURE_OPENAI_ENDPOINT")
OPENAI_KEY = st.secrets.get("AZURE_OPENAI_KEY")
OPENAI_DEPLOYMENT = st.secrets.get("AZURE_OPENAI_DEPLOYMENT_NAME")
OPENAI_VERSION = st.secrets.get("AZURE_OPENAI_API_VERSION", "2023-05-15")
# Initialize Cosmos client

cosmos_client = CosmosClient(COSMOS_ENDPOINT, credential=DefaultAzureCredential())
database = cosmos_client.create_database_if_not_exists(id=COSMOS_DATABASE)

catalog_container = database.create_container_if_not_exists(
    id=COSMOS_CATALOG_CONTAINER,
    partition_key=PartitionKey(path="/category"),
    offer_throughput=400
)
apl_container = database.create_container_if_not_exists(
    id=COSMOS_APL_CONTAINER,
    partition_key=PartitionKey(path="/benefit_category"),
    offer_throughput=400
)

# Initialize OpenAI client
openai_client = AzureOpenAI(
    api_key=OPENAI_KEY,  
    api_version=OPENAI_VERSION,
    azure_endpoint=OPENAI_ENDPOINT,
)

# --------------- Approved Taxonomy (25 Categories) ---------------
CATEGORIES = [
    "Beverages", "Snacks", "Dairy", "Produce", "Bakery", "Meat", "Seafood", "Frozen Foods", "Canned Goods",
    "Condiments", "Cleaning Supplies", "Personal Care", "Electronics", "Office Supplies", "Apparel", "Footwear",
    "Household Goods", "Pet Supplies", "Toys", "Automotive", "Pharmacy", "Health & Wellness", "Beauty", "Hardware", "Garden", "Miscellaneous"
]

FIELDS = [
    "sku", "upc", "plu", "vendor", "product_name",
    "product_description", "unit_of_measure", "category"
]

APL_FIELDS = [
    "benefit_category", "active", "updated_at", "sku", "category", "vendor"
]

# --------------- Helper Functions ---------------

def normalize_product_data(df: pd.DataFrame) -> pd.DataFrame:
    # Standardize casing
    df = df.copy()
    if "product_name" in df.columns:
        df["product_name"] = df["product_name"].astype(str).str.strip().str.title()
    # Normalize unit of measure
    if "unit_of_measure" in df.columns:
        df["unit_of_measure"] = df["unit_of_measure"].astype(str).str.strip().str.upper()
    # Add vendor field uniformity
    if "vendor" in df.columns:
        df["vendor"] = df["vendor"].astype(str).str.strip().str.title()
    # Drop exact duplicates based on SKU/UPC/PLU
    df = df.drop_duplicates(subset=[col for col in ["sku", "upc", "plu"] if col in df.columns], keep="first")

    for _, row in df.iterrows():
        product_name = row.get("product_name", "")
        description = row.get("description", "")
        category = classify_product(product_name, description)
        # Update the category for the row
        df.at[_, "category"] = category

    return df

def classify_product(product_name: str, description: str = None) -> str:
    """
    Ask the LLM to assign one of the predefined CATEGORIES to this product.
    Falls back to 'Miscellaneous' if the response isn't an exact match.
    """
    system_msg = (
        "You are a product taxonomy assistant. "
        "Given a product name and optional description, select exactly one category "
        "from the following list. Respond with only the category name."
    )
    user_msg = f"Categories: {', '.join(CATEGORIES)}\nProduct: {product_name}"
    if description:
        user_msg += f"\nDescription: {description}"

    resp = openai_client.chat.completions.create(
        model=OPENAI_DEPLOYMENT,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": user_msg}
        ],
        max_tokens=10,
        temperature=0.0  # deterministic
    )

    raw = resp.choices[0].message.content.strip().title()

    # clean out any backticks or extra punctuation
    raw = raw.strip("`\"' ")

    print(f"LLM classified product '{product_name}' as: {raw}")
    # validate against our known list
    if raw in CATEGORIES:
        return raw
    else:
        return "Miscellaneous"

def upsert_catalog_records(df: pd.DataFrame):
    print("Upserting catalog records...")
    try:
        for _, row in df.iterrows():
            prod_id = str(uuid.uuid4())
            record = {
                "id": prod_id,
                "sku": row.get("sku"),
                "upc": row.get("upc"),
                "plu": row.get("plu"),
                "product_name": row.get("product_name", ""),
                "description": row.get("description", ""),
                "unit_of_measure": row.get("unit_of_measure"),
                "vendor": row.get("vendor"),
                "category": row.get("category", "")
            }
            print(f"Upserting record: {record}")
            catalog_container.upsert_item(record)
    except Exception as e:
        st.error(f"Error upserting catalog records: {e}")
        print(f"Error upserting catalog records: {e}")
        raise e

def upsert_apl_records(df: pd.DataFrame):
    print("Upserting APL records...")
    if df.empty:
        print("No APL records to upsert.")
        return

    for _, row in df.iterrows():
        # Use natural key to make id deterministic for true upsert
        rule_id = f"{row['sku']}|{row['category']}|{row['vendor']}"
        record = {
            "id": rule_id,
            "benefit_category": row.get("category"),
            "active": row.get("active", True),
            "updated_at": row.get("updated_at", pd.Timestamp.now().isoformat()),
            "sku": row.get("sku"),
            "category": row.get("category"),
            "vendor": row.get("vendor"),
        }
        apl_container.upsert_item(record)

def query_cosmos(container, query: str, parameters: list = None) -> list:
    items = list(container.query_items(query=query, parameters=parameters or [], enable_cross_partition_query=True))
    return items

def generate_sample_apl(size: int = 5) -> pd.DataFrame:
    """
    Generate sample APL rules by randomly picking SKU, category, and vendor
    from the existing ProductCatalog stored in Cosmos DB.
    """
    # Query all products from Cosmos DB
    df_catalog = pd.DataFrame(query_cosmos(catalog_container, "SELECT * FROM c"))
    
    # If no data is found, return empty DataFrame
    if df_catalog.empty or not all(col in df_catalog.columns for col in ["sku", "category", "vendor"]):
        return pd.DataFrame(columns=["sku", "category", "vendor"])
    
    # Sample product rows
    sample_rows = df_catalog.sample(n=min(size, len(df_catalog)))[["sku", "category", "vendor"]]
    
    # Build rules
    sample_rules = []
    for sku, category, vendor in sample_rows.itertuples(index=False):
        sample_rules.append({
            "sku": sku,
            "category": category,
            "vendor": vendor,
            "active": True,
            "updated_at": pd.Timestamp.now().isoformat(),
        })
    
    return pd.DataFrame(sample_rules)

def generate_sample_catalog(vendor_name: str, start_id: int, size: int = 10, model: str = "gpt-4o-mini") -> pd.DataFrame:
    """
    Generates a sample product catalog by calling an LLM to create realistic
    product names and descriptions for random categories.
    """
    sample_records = []
    for i in range(start_id, start_id + size):
        # pick a random category
        category = random.choice(CATEGORIES)
        
        # build prompt asking LLM to return JSON with product_name & description
        prompt = (
            f"You are a product marketing assistant. "
            f"Generate a unique product for the category '{category}' sold by '{vendor_name}'. "
            f"Respond ONLY with a JSON object containing exactly two fields: "
            f"'product_name' and 'description'."
        )
        
        # call the OpenAI API
        resp = openai_client.chat.completions.create(
                model=OPENAI_DEPLOYMENT,
                messages=[{"role": "system", "content": "You generate product entries."},
                        {"role": "user",   "content": prompt}],
                temperature=0.8,
                max_tokens=1000
            )
        
        # parse the JSON from the LLM response
        raw  = resp.choices[0].message.content.strip()
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start != -1 and end != -1:
            candidate = raw[start:end]
            try:
                entry = json.loads(candidate)
                #print(f"Parsed JSON: {entry}")
                # override defaults only if keys exist
                if "product_name" in entry:
                    product_name = entry["product_name"]
                if "description" in entry:
                    description  = entry["description"]
            except json.JSONDecodeError:
                # parsing failed, keep safe defaults
                # fallback if parsing fails
                #product_name = f"{vendor_name} {category} Item {i}"
                #description  = f"High-quality {category.lower()} item from {vendor_name}."
                print(f"Failed to parse LLM response: {raw}")
        
        # generate the rest of the fields
        sku   = f"{vendor_name[:3].upper()}-{i:04d}"
        upc   = str(random.randint(100_000_000_000, 999_999_999_999))
        plu   = str(1000 + i)
        uom   = random.choice(["Each", "Pack", "Box", "Kg", "Liter"]).upper()
        
        sample_records.append({
            "sku":             sku,
            "upc":             upc,
            "plu":             plu,
            "vendor":          vendor_name,
            #"category":        category,
            "product_name":    product_name,
            "description":     description,
            "unit_of_measure": uom,
        })
    
    return pd.DataFrame(sample_records)

def parse_filter(chat_input: str) -> dict:
    """
    Ask the LLM to extract a single field & value filter
    from the user’s natural-language request.
    Expects a JSON response: {"field": "...", "value": "..."}
    """
    system = (
        "You are a query parser for a product catalog. "
        "The catalog fields are: " + ", ".join(FIELDS) + ".\n"
        "Given a user request, pick exactly one field to filter on, "
        "and return a JSON object with keys 'field' and 'value'.\n"
        "If the request does not specify any valid filter, respond with {}."
    )
    user = f"User asked: \"{chat_input}\""

    resp = openai_client.chat.completions.create(
                model=OPENAI_DEPLOYMENT,
                messages=[
            {"role": "system",  "content": system},
            {"role": "user",    "content": user}
        ],
        temperature=0.0,
        max_tokens=50
            )
    print(f"LLM response: {resp.choices[0].message.content.strip()}")
    raw  = resp.choices[0].message.content.strip()
    start = raw.find("{")
    end   = raw.rfind("}") + 1
    if start != -1 and end != -1:
        candidate = raw[start:end]
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            print(f"Failed to parse filter JSON: {raw}")
            return {}
    # sanitize
    field = parsed.get("field", "").lower()
    value = parsed.get("value", "").strip()
    if field in FIELDS and value:
        return {"field": field, "value": value}
    return {}

def parse_apl_filter(chat_input: str) -> dict:
    """
    Ask the LLM to extract a single field & value filter
    from the user’s natural-language request.
    Expects a JSON response: {"field": "...", "value": "..."}
    """
    system = (
        "You are a query parser for a product catalog. "
        "The catalog fields are: " + ", ".join(APL_FIELDS) + ".\n"
        "Given a user request, pick exactly one field to filter on, "
        "and return a JSON object with keys 'field' and 'value'.\n"
        "If the request does not specify any valid filter, respond with {}."
    )
    user = f"User asked: \"{chat_input}\""

    resp = openai_client.chat.completions.create(
                model=OPENAI_DEPLOYMENT,
                messages=[
            {"role": "system",  "content": system},
            {"role": "user",    "content": user}
        ],
        temperature=0.0,
        max_tokens=50
            )
    print(f"LLM response: {resp.choices[0].message.content.strip()}")
    raw  = resp.choices[0].message.content.strip()
    start = raw.find("{")
    end   = raw.rfind("}") + 1
    if start != -1 and end != -1:
        candidate = raw[start:end]
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            print(f"Failed to parse filter JSON: {raw}")
            return {}
    # sanitize
    field = parsed.get("field", "").lower()
    value = parsed.get("value", "").strip()
    if field in FIELDS and value:
        return {"field": field, "value": value}
    return {}

# def generate_sample_catalog(vendor_name: str, start_id: int, size: int = 10) -> pd.DataFrame:
#     categories = random.sample(CATEGORIES, min(len(CATEGORIES), size))
#     sample_records = []
#     for i in range(start_id, start_id + size):
#         prod_base = random.choice(["widget", "gadget", "item", "product"]).capitalize()
#         variation = random.choice(["Pro", "Max", "Lite", "XL", "Mini"])
#         product_name = f"{vendor_name} {prod_base} {variation} {i}"
#         description = f"High-quality {prod_base.lower()} from {vendor_name}."
#         sku = f"{vendor_name[:3].upper()}-{i:04d}"
#         upc = f"{random.randint(100000000000, 999999999999)}"
#         plu = f"{1000 + i}"
#         uom = random.choice(["Each", "Pack", "Box", "Kg", "Liter"]).upper()
#         sample_records.append({
#             "sku": sku,
#             "upc": upc,
#             "plu": plu,
#             "product_name": product_name,
#             "description": description,
#             "unit_of_measure": uom,
#             "vendor": vendor_name
#         })
#     return pd.DataFrame(sample_records)

# --------------- Streamlit App ---------------
st.set_page_config(page_title="Real-Time Spend Approval POC", layout="wide")
st.title("🛍️ Real-Time Spend Approval POC")

tabs = st.tabs(["Upload Data", "Chat Interface", "Transaction Simulation"])

# --------------- Tab 1: Upload Data ---------------
with tabs[0]:
    st.header("Upload or Pull Product Catalog & APL Rules")
    st.markdown("You can upload CSVs or pull catalogs from multiple vendors. All data will be normalized, enriched, and classified into 25 customer-approved categories.")

    # Section to generate sample data for multiple vendors
    st.subheader("Pull Product Catalog from Vendors")
    cols = st.columns([1, 1, 1])
    vendor1 = cols[0].text_input("Vendor 1 Name", "Walmart")
    vendor2 = cols[1].text_input("Vendor 2 Name", "JewelOsco")
    vendor3 = cols[2].text_input("Vendor 3 Name", "Marianos")
    if st.button("Pull Product Catalog", key="gen_sample"):
        df_vendor1 = generate_sample_catalog(vendor1, start_id=1)
        df_vendor2 = generate_sample_catalog(vendor2, start_id=101)
        df_vendor3 = generate_sample_catalog(vendor3, start_id=201)
        df_all = pd.concat([df_vendor1, df_vendor2, df_vendor3], ignore_index=True)
        st.subheader("Preview - Raw Combined Product Catalog")
        st.dataframe(df_all)
        df_clean = normalize_product_data(df_all)
        st.subheader("Preview - Cleaned Product Catalog")
        st.dataframe(df_clean)
        # Allow download of the cleaned sample catalog
        csv = df_clean.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Cleaned Product Catalog",
            data=csv,
            file_name="cleaned_product_catalog.csv",
            mime="text/csv",
            key="download_product_catalog"
        )

        if st.button("Ingest Product Catalog into Cosmos DB", key="ingest_sample"):
            try:
                with st.spinner("Ingesting and Classifying Products..."):
                    upsert_catalog_records(df_clean)
                st.success("Vendor product catalog ingested successfully!")
            except Exception as e:
                st.error(f"Error ingesting Product catalog: {e}")
                print(f"Error ingesting Product catalog: {e}")

    st.divider()

    st.subheader("Upload Product Catalog CSV")
    upload_catalog = st.file_uploader("Upload Product Catalog CSV", type=["csv"], key="catalog_uploader")
    if upload_catalog:
        df_catalog = pd.read_csv(upload_catalog)
        st.subheader("Preview - Raw Product Catalog")
        st.dataframe(df_catalog.head())
        df_clean = normalize_product_data(df_catalog)
        st.subheader("Preview - Cleaned Product Catalog")
        st.dataframe(df_clean.head())
        if st.button("Ingest Product Catalog", key="ingest_upload"):
            with st.spinner("Ingesting and Classifying Products..."):
                upsert_catalog_records(df_clean)
            st.success("Product catalog ingested successfully!")

    st.divider()

    st.subheader("Retrieve All Products from Catalog")
    if st.button("Retrieve Product", key="retrieve_product"):
        query = "SELECT * FROM c"
        items = query_cosmos(catalog_container, query)
        if items:
            df_all_products = pd.DataFrame(items)
            st.subheader("All Products in Catalog")
            st.dataframe(df_all_products)
        else:
            st.info("No products found in catalog.")
            
    st.divider()

    # Sample APL Generation
    st.subheader("APL Rules")
    apl_size = st.number_input("Number of sample APL rules", min_value=1, max_value=20, value=5)
    if st.button("APL", key="gen_apl"):
        df_apl_sample = generate_sample_apl(size=apl_size)
        st.subheader("Preview - Sample APL Rules")
        st.dataframe(df_apl_sample)
        st.session_state["apl_sample"] = df_apl_sample
    if "apl_sample" in st.session_state:
        if st.button("Ingest APL Rules", key="ingest_apl_sample"):
            print("Ingesting APL rules...")
            try:
                with st.spinner("Ingesting APL Rules..."):
                    upsert_apl_records(st.session_state["apl_sample"])
                st.success("APL rules ingested successfully!")
            except Exception as e:
                st.error(f"Error ingesting sample APL rules: {e}")
                print(f"Error ingesting sample APL rules: {e}")
            
    st.divider()

    st.subheader("Upload APL Rules CSV")
    upload_apl = st.file_uploader("Upload APL Rules CSV", type=["csv"], key="apl_uploader")
    if upload_apl:
        df_apl = pd.read_csv(upload_apl)
        st.subheader("Preview - Raw APL Rules")
        st.dataframe(df_apl.head())
        if st.button("Ingest APL Rules", key="ingest_apl_upload"):
            with st.spinner("Ingesting APL Rules..."):
                upsert_apl_records(df_apl)
            st.success("APL rules ingested successfully!")

# --------------- Tab 2: Chat Interface ---------------
with tabs[1]:
    st.header("Chat Interface for Catalog & APL Management")
    st.markdown("Use natural language to query or manage the catalog and APL rules.")

    chat_input = st.text_input("Enter your query or command:")
    if st.button("Send", key="chat_send") and chat_input:
        if "add apl" in chat_input.lower() or "insert apl" in chat_input.lower():
            prompt = f"Extract vendor, sku, category, benefit_category from: {chat_input}. benefit_category is same as category. Return as JSON."
            response = openai_client.chat.completions.create(
                model=OPENAI_DEPLOYMENT,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150
            )
            content = response.choices[0].message.content.strip()
            start = content.find("{")
            end   = content.rfind("}") + 1
            if start != -1 and end != -1:
                candidate = content[start:end]
                try:
                    rule_json = json.loads(candidate)
                    print(f"Parsed APL rule JSON: {rule_json}")
                    rule_df = pd.DataFrame([rule_json])
                    upsert_apl_records(rule_df)
                    st.success(f"APL rule added: {rule_json}")
                except Exception as e:
                    st.error(f"Failed to parse APL command: {e}")
        else:
            # Determine intent: catalog vs. APL
            lowered = chat_input.lower()
            if "catalog" in lowered:
                # # e.g., "Show products in category Beverages"
                # if "category" in lowered:
                #     category = chat_input.split("category")[-1].strip().title()
                #     if category in CATEGORIES:
                #         query = "SELECT * FROM c WHERE c.category = @category"
                #         items = query_cosmos(catalog_container, query, parameters=[{"name": "@category", "value": category}])
                #         if items:
                #             st.dataframe(pd.DataFrame(items))
                #         else:
                #             st.info("No products found for that category.")
                #     else:
                #         st.info(f"Category must be one of: {', '.join(CATEGORIES)}")
                # else:
                #     st.info("Please specify 'category <CategoryName>'.")
                filt = parse_filter(chat_input)
                if filt:
                    field, val = filt["field"], filt["value"]
                    query = f"SELECT * FROM c WHERE c.{field} = @val"
                    items = query_cosmos(
                        catalog_container,
                        query,
                        parameters=[{"name":"@val", "value": val}]
                    )
                    if items:
                        st.dataframe(pd.DataFrame(items))
                    else:
                        st.info(f"No products found where {field} = '{val}'.")
                else:
                    st.info(
                        "Sorry, I couldn’t figure out what to filter on.  "
                        "Try something like “Show catalog items in category Dairy” "
                        "or “Show catalog items from vendor AcmeCo.”"
                    )
            elif "apl" in lowered or "allowed product" in lowered:
                # if "for" in lowered:
                #     benefit = chat_input.split("for")[-1].strip().title()
                #     query = "SELECT * FROM c WHERE c.benefit_category = @benefit"
                #     items = query_cosmos(apl_container, query, parameters=[{"name": "@benefit", "value": benefit}])
                #     if items:
                #         st.dataframe(pd.DataFrame(items))
                #     else:
                #         st.info("No APL rules found for that benefit category.")
                # else:
                #     st.info("Please specify 'APL for <BenefitCategory>'.")
                filt = parse_apl_filter(chat_input)
                if filt:
                    field, val = filt["field"], filt["value"]
                    query = f"SELECT * FROM c WHERE c.{field} = @val"
                    items = query_cosmos(
                        apl_container,
                        query,
                        parameters=[{"name":"@val", "value": val}]
                    )
                    if items:
                        st.dataframe(pd.DataFrame(items))
                    else:
                        st.info(f"No products found where {field} = '{val}'.")
                else:
                    st.info(
                        "Sorry, I couldn’t figure out what to filter on.  "
                        "Try something like “Show catalog items in category Dairy” "
                        "or “Show catalog items from vendor AcmeCo.”"
                    )
            else:
                st.info("Specify whether you want to query 'catalog' or 'APL'.")

# --------------- Tab 3: Transaction Simulation ---------------
with tabs[2]:
    st.header("Transaction Simulation & Spend Approval")
    st.markdown("Generate synthetic item-level spend and run approval logic against the APL rules.")

    if st.button("Generate Synthetic Transactions", key="gen_tx"):
        sample_data = {
            "transaction_id": [str(uuid.uuid4()) for _ in range(10)],
            "sku": [f"SKU{num:03d}" for num in range(10)],
            "product_name": [random.choice([f"{random.choice(CATEGORIES)} Item {i}" for i in range(10)]) for _ in range(10)],
            "unit_price": [round(5 + num * 0.5, 2) for num in range(10)],
            "quantity": [random.randint(1, 5) for _ in range(10)]
        }
        df_transactions = pd.DataFrame(sample_data)
        st.subheader("Synthetic Transactions")
        st.dataframe(df_transactions)
        st.session_state["transactions"] = df_transactions

    if "transactions" in st.session_state:
        df_tx = st.session_state["transactions"].copy()
        if st.button("Run Spend Approval", key="run_approval"):
            results = []
            for _, row in df_tx.iterrows():
                product_name = row["product_name"]
                category = classify_product(product_name)
                query = "SELECT * FROM c WHERE c.benefit_category = @benefit"
                rules = query_cosmos(apl_container, query, parameters=[{"name": "@benefit", "value": category}])
                approved = False
                reason = "Not Eligible"
                if rules:
                    for rule in rules:
                        keywords = [kw.strip().lower() for kw in rule.get("eligible_keywords", "").split(",")]
                        if any(kw in product_name.lower() for kw in keywords):
                            approved = True
                            reason = "Approved"
                            break
                results.append({
                    "transaction_id": row["transaction_id"],
                    "product_name": product_name,
                    "category": category,
                    "approved": approved,
                    "reason": reason
                })
            df_results = pd.DataFrame(results)
            st.subheader("Approval Results")
            st.dataframe(df_results)
