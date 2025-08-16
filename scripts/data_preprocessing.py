"""Script for data preprocessing"""

import pandas as pd
import logging
from pathlib import Path

def preprocess_data():
    """Preprocess training data"""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    data_dir = Path('data')
    
    # Load data files
    findings_file = data_dir / 'findings_with_roles.xlsx'
    keywords_file = data_dir / 'intern2task.xlsx'
    
    if findings_file.exists():
        df = pd.read_excel(findings_file)
        logger.info(f"Loaded {len(df)} findings")
        
        # Basic preprocessing
        df = df.dropna()
        df['Finding'] = df['Finding'].str.strip()
        
        # Save preprocessed data
        output_file = data_dir / 'preprocessed_findings.xlsx'
        df.to_excel(output_file, index=False)
        logger.info(f"Preprocessed data saved to {output_file}")
    
    else:
        logger.warning(f"Data file not found: {findings_file}")

if __name__ == '__main__':
    preprocess_data()
