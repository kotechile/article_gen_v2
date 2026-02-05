#!/usr/bin/env python3
"""
Script to add the latest LLM models to an existing database
Run this after the initial migration to add new models
"""

import os
import sys
import asyncio
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.llm_config import LLMProvider

# Database URL from environment
DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///./trendtap.db')

def add_latest_models():
    """Add the latest LLM models to the database"""
    
    # Create database connection
    engine = create_engine(DATABASE_URL)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionLocal()
    
    try:
        # Define the latest models
        latest_models = [
            # OpenAI Latest
            {
                'name': 'OpenAI GPT-5 Mini',
                'provider_type': 'openai',
                'model_name': 'gpt-5-mini',
                'api_key_env_var': 'OPENAI_API_KEY',
                'max_tokens': 4000,
                'temperature': 0.7,
                'cost_per_1k_tokens': 0.0015,
                'is_default': True,  # Set as new default
                'priority': 120,
                'is_active': True
            },
            # Anthropic Latest
            {
                'name': 'Anthropic Claude 3.5 Sonnet',
                'provider_type': 'anthropic',
                'model_name': 'claude-3-5-sonnet-20241022',
                'api_key_env_var': 'ANTHROPIC_API_KEY',
                'max_tokens': 4000,
                'temperature': 0.7,
                'cost_per_1k_tokens': 0.003,
                'is_default': False,
                'priority': 95,
                'is_active': True
            },
            # Google Latest
            {
                'name': 'Google Gemini 2.5 Flash',
                'provider_type': 'google',
                'model_name': 'gemini-2.5-flash',
                'api_key_env_var': 'GOOGLE_API_KEY',
                'max_tokens': 4000,
                'temperature': 0.7,
                'cost_per_1k_tokens': 0.0005,
                'is_default': False,
                'priority': 85,
                'is_active': True
            },
            {
                'name': 'Google Gemini 2.5 Flash Lite',
                'provider_type': 'google',
                'model_name': 'gemini-2.5-flash-lite',
                'api_key_env_var': 'GOOGLE_API_KEY',
                'max_tokens': 4000,
                'temperature': 0.7,
                'cost_per_1k_tokens': 0.0001,
                'is_default': False,
                'priority': 70,
                'is_active': True
            },
            # Local Latest
            {
                'name': 'Local Ollama Llama3.1',
                'provider_type': 'local',
                'model_name': 'llama3.1',
                'api_key_env_var': None,
                'base_url': 'http://localhost:11434',
                'max_tokens': 2000,
                'temperature': 0.7,
                'cost_per_1k_tokens': 0.0,
                'is_default': False,
                'priority': 60,
                'is_active': False  # Disabled by default
            }
        ]
        
        # Check if models already exist and add new ones
        added_count = 0
        for model_data in latest_models:
            existing = db.query(LLMProvider).filter(
                LLMProvider.name == model_data['name']
            ).first()
            
            if not existing:
                # Remove default from all other providers if this is the new default
                if model_data.get('is_default', False):
                    db.query(LLMProvider).update({'is_default': False})
                
                # Create new provider
                provider = LLMProvider(**model_data)
                db.add(provider)
                added_count += 1
                print(f"✅ Added: {model_data['name']}")
            else:
                print(f"⚠️  Already exists: {model_data['name']}")
        
        # Commit changes
        db.commit()
        print(f"\n🎉 Successfully added {added_count} new models!")
        
        # Show current providers
        print("\n📋 Current LLM Providers:")
        providers = db.query(LLMProvider).filter(LLMProvider.is_active == True).order_by(LLMProvider.priority.desc()).all()
        for provider in providers:
            status = "🔥 DEFAULT" if provider.is_default else "✅ Active"
            print(f"  {status} {provider.name} ({provider.provider_type}) - ${provider.cost_per_1k_tokens}/1K tokens")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        db.rollback()
        return False
    finally:
        db.close()
    
    return True

def update_existing_models():
    """Update existing models with new capabilities"""
    
    engine = create_engine(DATABASE_URL)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionLocal()
    
    try:
        # Update GPT-4 Turbo with higher token limit
        gpt4_turbo = db.query(LLMProvider).filter(
            LLMProvider.name == 'OpenAI GPT-4 Turbo'
        ).first()
        
        if gpt4_turbo:
            gpt4_turbo.max_tokens = 4000
            gpt4_turbo.cost_per_1k_tokens = 0.01
            print("✅ Updated GPT-4 Turbo with 4K token limit")
        
        # Update Claude 3.5 Sonnet if it exists
        claude_35 = db.query(LLMProvider).filter(
            LLMProvider.name == 'Anthropic Claude 3.5 Sonnet'
        ).first()
        
        if claude_35:
            claude_35.max_tokens = 4000
            claude_35.cost_per_1k_tokens = 0.003
            print("✅ Updated Claude 3.5 Sonnet with 4K token limit")
        
        db.commit()
        print("✅ Updated existing models with new capabilities")
        
    except Exception as e:
        print(f"❌ Error updating models: {str(e)}")
        db.rollback()
        return False
    finally:
        db.close()
    
    return True

if __name__ == "__main__":
    print("🚀 Adding Latest LLM Models to TrendTap")
    print("=" * 50)
    
    # Add new models
    if add_latest_models():
        print("\n🔄 Updating existing models...")
        update_existing_models()
        
        print("\n✨ Setup complete!")
        print("\nNext steps:")
        print("1. Set your API keys in environment variables")
        print("2. Test the new models in the admin interface")
        print("3. Set your preferred model as default")
        print("\nAPI Keys needed:")
        print("- OPENAI_API_KEY for GPT-5 Mini")
        print("- ANTHROPIC_API_KEY for Claude 3.5 Sonnet")
        print("- GOOGLE_API_KEY for Gemini 2.5 Flash")
    else:
        print("\n❌ Setup failed. Please check the error messages above.")
        sys.exit(1)


