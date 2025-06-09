## Environment Configuration

Create .streamlitfolder and \secrets.toml file with following configuration data

### CosmosDB Config
- COSMOSDB_ENDPOINT         = ""
- COSMOSDB_DATABASE         = "semantickernel"
- COSMOS_CATALOG_CONTAINER  = "ProductCatalog"
- COSMOS_APL_CONTAINER      = "AllowedProductList"

### Azure OpenAI Config
- AZURE_OPENAI_ENDPOINT = ""
- AZURE_OPENAI_DEPLOYMENT_NAME = "chat4o"
- AZURE_OPENAI_API_VERSION = "2024-08-01-preview"
- AZURE_OPENAI_KEY = ""
- AZURE_OPENAI_MODEL_NAME = "gpt-4o"

## Sample Query
- Show catalog items from vendor Walmart
- Show products in catalog for category Bakery
- add apl for vendor Marianos on SKU MRNO-0003 for Footwear
- show apl for vendor Marianos
- show apl for category Footwear
