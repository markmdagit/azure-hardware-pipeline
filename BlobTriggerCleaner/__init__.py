import logging
import azure.functions as func
from pipeline.cleaner import run_pipeline

def main(myblob: func.InputStream, outputblob: func.Out[bytes]):
    logging.info(f"Blob trigger received file: {myblob.name} ({myblob.length} bytes)")

    try:
        raw_data = myblob.read()
        parquet_data = run_pipeline(raw_data)
        outputblob.set(parquet_data)
        logging.info("Processing complete. Cleaned Parquet written to clean-logs.")
    except Exception as e:
        logging.error(f"Error processing blob: {str(e)}")
        raise e
