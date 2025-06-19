from langchain_aws import BedrockEmbeddings

def get_embedding_function():
    embeddings = BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v2:0",
        credentials_profile_name="default",
        region_name="ap-south-1"
    )
    return embeddings