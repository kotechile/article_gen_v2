"""
Upload API Endpoints

Handles file upload, validation, and processing for Ahrefs keyword export files.
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional
import logging
import uuid
from datetime import datetime
import os
from pathlib import Path

from ..models.ahrefs_export_file import AhrefsExportFile
from ..services.file_parser import FileParser
from ..services.database import DatabaseService
from ..utils.validation import ValidationUtility

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/upload", tags=["upload"])

# Initialize services
file_parser = FileParser()
database = DatabaseService()
validator = ValidationUtility()

# In-memory storage for file processing status (in production, use Redis or database)
file_status = {}

@router.post("/")
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    user_id: str = None  # This would come from authentication middleware
) -> Dict[str, Any]:
    """
    Upload and process Ahrefs keyword export file
    
    Args:
        background_tasks: FastAPI background tasks
        file: Uploaded file
        user_id: Authenticated user ID
        
    Returns:
        Upload response with file_id and status
    """
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Check file size
        file_content = await file.read()
        file_size = len(file_content)
        
        # Create temporary file path
        file_id = str(uuid.uuid4())
        temp_dir = Path("temp_uploads")
        temp_dir.mkdir(exist_ok=True)
        temp_file_path = temp_dir / f"{file_id}.tsv"
        
        # Save file temporarily
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(file_content)
        
        # Validate file upload
        is_valid, errors = validator.validate_file_upload(
            str(temp_file_path), 
            file_size, 
            file.content_type
        )
        
        if not is_valid:
            # Clean up temp file
            temp_file_path.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail=f"File validation failed: {', '.join(errors)}")
        
        # Validate TSV format
        is_tsv_valid, tsv_errors, dataframe = validator.validate_tsv_format(str(temp_file_path))
        
        if not is_tsv_valid:
            # Clean up temp file
            temp_file_path.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail=f"TSV validation failed: {', '.join(tsv_errors)}")
        
        # Create file record
        ahrefs_file = AhrefsExportFile(
            id=file_id,
            filename=file.filename,
            file_size=file_size,
            file_path=str(temp_file_path),
            user_id=user_id,
            status="uploaded",
            created_at=datetime.utcnow()
        )
        
        # Store file record in database
        await database.save_ahrefs_file(ahrefs_file)
        
        # Initialize processing status
        file_status[file_id] = {
            "status": "uploaded",
            "progress": 0,
            "message": "File uploaded successfully",
            "created_at": datetime.utcnow()
        }
        
        # Start background processing
        background_tasks.add_task(process_uploaded_file, file_id, str(temp_file_path))
        
        logger.info(f"File uploaded successfully: {file_id}")
        
        return {
            "file_id": file_id,
            "status": "uploaded",
            "message": "File uploaded successfully",
            "filename": file.filename,
            "file_size": file_size,
            "created_at": ahrefs_file.created_at.isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@router.get("/{file_id}/status")
async def get_upload_status(file_id: str) -> Dict[str, Any]:
    """
    Get upload and processing status for a file
    
    Args:
        file_id: Unique file identifier
        
    Returns:
        Status information
    """
    try:
        # Check if file exists in status tracking
        if file_id not in file_status:
            raise HTTPException(status_code=404, detail="File not found")
        
        status_info = file_status[file_id]
        
        # Get file record from database
        ahrefs_file = await database.get_ahrefs_file(file_id)
        if not ahrefs_file:
            raise HTTPException(status_code=404, detail="File record not found")
        
        return {
            "file_id": file_id,
            "status": status_info["status"],
            "progress": status_info["progress"],
            "message": status_info["message"],
            "filename": ahrefs_file.filename,
            "file_size": ahrefs_file.file_size,
            "created_at": ahrefs_file.created_at.isoformat(),
            "updated_at": status_info["created_at"].isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting upload status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")

async def process_uploaded_file(file_id: str, file_path: str) -> None:
    """
    Background task to process uploaded file
    
    Args:
        file_id: Unique file identifier
        file_path: Path to the uploaded file
    """
    try:
        logger.info(f"Starting file processing: {file_id}")
        
        # Update status
        file_status[file_id].update({
            "status": "processing",
            "progress": 10,
            "message": "Parsing file content"
        })
        
        # Parse file content
        keywords = await file_parser.parse_ahrefs_file(file_path)
        
        # Update status
        file_status[file_id].update({
            "status": "processing",
            "progress": 50,
            "message": f"Parsed {len(keywords)} keywords"
        })
        
        # Save keywords to database
        await database.save_keywords(file_id, keywords)
        
        # Update status
        file_status[file_id].update({
            "status": "completed",
            "progress": 100,
            "message": f"Successfully processed {len(keywords)} keywords",
            "keywords_count": len(keywords)
        })
        
        # Update file record
        ahrefs_file = await database.get_ahrefs_file(file_id)
        if ahrefs_file:
            ahrefs_file.status = "completed"
            ahrefs_file.keywords_count = len(keywords)
            ahrefs_file.processed_at = datetime.utcnow()
            await database.update_ahrefs_file(ahrefs_file)
        
        logger.info(f"File processing completed: {file_id}")
        
    except Exception as e:
        logger.error(f"Error processing file {file_id}: {str(e)}")
        
        # Update status with error
        file_status[file_id].update({
            "status": "error",
            "progress": 0,
            "message": f"Processing failed: {str(e)}"
        })
        
        # Update file record
        ahrefs_file = await database.get_ahrefs_file(file_id)
        if ahrefs_file:
            ahrefs_file.status = "error"
            ahrefs_file.error_message = str(e)
            await database.update_ahrefs_file(ahrefs_file)

@router.delete("/{file_id}")
async def delete_file(file_id: str) -> Dict[str, Any]:
    """
    Delete uploaded file and its data
    
    Args:
        file_id: Unique file identifier
        
    Returns:
        Deletion confirmation
    """
    try:
        # Get file record
        ahrefs_file = await database.get_ahrefs_file(file_id)
        if not ahrefs_file:
            raise HTTPException(status_code=404, detail="File not found")
        
        # Delete physical file
        if ahrefs_file.file_path and os.path.exists(ahrefs_file.file_path):
            os.remove(ahrefs_file.file_path)
        
        # Delete from database
        await database.delete_ahrefs_file(file_id)
        
        # Remove from status tracking
        if file_id in file_status:
            del file_status[file_id]
        
        logger.info(f"File deleted: {file_id}")
        
        return {
            "file_id": file_id,
            "status": "deleted",
            "message": "File and associated data deleted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Deletion failed: {str(e)}")

