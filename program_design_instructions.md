
The existing code needs to be refactored to create a new, clean and efficient article generation program.


  1. Program Overview and Article Generation Process

  Create an improved and reliable content generation pipeline that transforms a user-provided research brief into a comprehensive, well-structured, and cited article. This program needs to produce high-quality, SEO-optimized 
  content suitable for platforms like WordPress.

  The article generation process is a multi-stage, asynchronous pipeline orchestrated by Celery, which allows the system to handle long-running and computationally intensive tasks without blocking the main application. 

  Here's a step-by-step breakdown of the process:

   1. Request Initiation: A user submits a research request via a POST request to the /api/v1/research endpoint. This request includes a brief (the topic), desired keywords, and specifies the Large Language Model (LLM), keywords, llm_key, depth, rag_endpoint, tone, target_word_count , rag_collection_name and more. This is an example from the Noodl Application calling this program: 



        // Prepare the request payload
        const payload = {
            brief: Inputs.userOutline,
            claims_research_enabled: Inputs.claims_research_enabled,
            keywords: keywords,
            llm_model: Inputs.llm_model_agentic_researcher,
 //           llm_model: 'gemini/gemini-1.5-pro',
            llm_key: Inputs.llm_key,
            depth: Inputs.depth || "comprehensive",
            rag_endpoint: Inputs.rag_endpoint + '/query_hybrid_enhanced',
            tone: Inputs.tone || 'journalistic',
            target_word_count: parseInt(Inputs.target_word_count || 2000, 10)

        };

        // Add optional parameters if provided
        if (Inputs.rag_collection_name) {
            payload.rag_collection = Inputs.rag_collection_name;
            payload.rag_enabled = true;
            } else {
            payload.rag_enabled = false;          
            }


        if (Inputs.rag_llm_provider) payload.rag_llm_provider = Inputs.rag_llm_provider;

        console.log('Starting research with payload:', JSON.stringify(payload, null, 2));

        // Make the API call to start research
        const response = await fetch(Inputs.article_research_base_url + '/research', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-API-Key': Inputs.article_research_key
            },
            body: JSON.stringify(payload)
        });


   2. Task Queuing: The application creates a new research task, assigns it a unique task_id, and places it in a Celery queue for background processing. The API immediately returns the task_id to the user, who can then use 
      it to poll for status and results.

   3. Claim Extraction: The first step in the pipeline is to analyze the user's brief and extract key claims or questions that need to be addressed in the article. This is done by an LLM.

   4. Evidence Collection: For each extracted claim, the system gathers evidence from multiple sources:
       * Retrieval-Augmented Generation (RAG): If a rag_collection is specified, the system queries a private knowledge base to find relevant information.
       * Web Search (Linkup): The system uses the Linkup service as an option driven by input parameter "claims_research_enabled" to perform web searches, gathering up-to-date information from various online sources to support and expand upon the claims.

   5. Evidence Ranking: The collected evidence from both RAG and web search is then ranked based on its relevance to the claim, the credibility of the source, and overall quality. This ensures that the most reliable and 
      pertinent information is used in the article.

   6. Article Structure Generation: Based on the ranked evidence and the original brief, an LLM generates an optimal structure for the article, including a title, a hook to draw the reader in, an excerpt, a central thesis, 
      and a set of section titles.

   7. Content Generation: The system then proceeds to write the content for each section of the article. An LLM is used to generate the text, incorporating the ranked evidence and adhering to the specified tone and 
      target_word_count.

   8. Refinement and SEO Optimization: The generated article undergoes a refinement pipeline where a series of specialized "agents" (LLM-powered functions) review and improve the content. This includes:
       * Fact-checking: Verifying the accuracy of the information presented.
       * title to include keyword and not exceed 70 characters without truncation
       * SEO Optimization: Ensuring the article is optimized for search engines by including keywords, a meta description, and other SEO-related elements.
       * Clarity and Flow: Improving the readability and logical flow of the article.
       * Tone and Voice: Adjusting the writing style to match the desired tone.
       * Humanization: Making the text sound more natural and less like it was generated by an AI.

   9. Citation Generation: The system generates a list of formatted citations for all the evidence used in the article, ensuring proper attribution of sources.

   10. Final Output: The final, polished article is stored and made available to the user. The user can retrieve the completed article by making a GET request to the /api/v1/research/{task_id}/result endpoint.

  2. API Endpoints

  The application exposes the following API endpoints under the /api/v1 prefix:

   * `POST /research`: Creates a new research task.
       * Inputs (JSON body):
           * brief (string, required): A detailed description of the research topic.
           * provider (string, required): The LLM provider (e.g., 'openai', 'anthropic').
           * model (string, required): The model name (e.g., 'gpt-4o', 'claude-3.5-sonnet').
           * api_key (string, required): The API key for the LLM.
           * keywords (string, required): A comma-separated list of keywords.
           * depth (string, optional): The depth of the research ('standard', 'comprehensive', or 'deep'). Defaults to 'standard'.
           * tone (string, optional): The tone of the article ('academic', 'journalistic', etc.). Defaults to 'journalistic'.
           * target_word_count (integer, optional): The desired word count. Defaults to 2000.
           * rag_collection (string, optional): The name of the RAG collection to use.
           * rag_endpoint (string, optional): The URL of the RAG system.
           * rag_llm_provider (string, optional): The LLM provider for RAG queries.
       * Output (JSON):
           * A confirmation object containing the research_id (task ID), status ("accepted"), and other metadata about the task.

   * `GET /research/{task_id}`: Retrieves the status of a research task.
       * Inputs:
           * task_id (string, URL path): The ID of the research task.
       * Output (JSON):
           * An object containing the task_id, status ('PENDING', 'PROGRESS', 'SUCCESS', 'FAILURE'), progress percentage, current step, and other metadata.

   * `GET /research/{task_id}/result`: Retrieves the result of a completed research task.
       * Inputs:
           * task_id (string, URL path): The ID of the research task.
       * Output (JSON):
           * If the task is not yet complete, it returns an error with a 202 status code.
           * If the task has failed, it returns an error with details about the failure.
           * If the task is successful, it returns the complete generated article, including the title, content, sections, citations, and all associated metadata.

   * `POST /research/{task_id}/cancel`: Cancels a running research task.
       * Inputs:
           * task_id (string, URL path): The ID of the research task.
       * Output (JSON):
           * A confirmation message indicating that the task has been canceled.

   * `GET /health`: A health check endpoint to verify that the service is running.
       * Inputs: None.
       * Output (JSON):
           * A JSON object with status: 'healthy' and a timestamp.

  3. Tool Usage

   * Celery: Celery is the backbone of the asynchronous task processing system. It is used to manage a queue of research tasks and distribute them to background worker processes. This is crucial for handling the 
     long-running nature of the article generation process without blocking the API. The application defines a process_research_task in app/research_core/celery_tasks.py that encapsulates the entire article generation 
     pipeline. The run_celery_worker.py script is used to start the Celery workers that consume tasks from this queue.

   * LiteLLM: LiteLLM is used as a unified interface to interact with a variety of Large Language Models (LLMs) from different providers (like OpenAI, Anthropic, Google, etc.). The litellm_client.py module provides a 
     robust client that includes features like:
       * Model Routing: It can select the best model for a given task based on cost, speed, or quality.
       * Retry Logic: It automatically retries failed requests with exponential backoff.
       * Cost Tracking: It tracks the cost of LLM usage.
       * Fallback Mechanisms: If a request to one model fails, it can automatically fall back to another.
      The litellm_config.yaml file defines the available models and their configurations.
      Please note that latest gpt-5-mini and gemini-2.5-mini models need the temperature parameter removed to work.  grt 5 also needs max_completion_tokens: This parameter limits the combined total of reasoning and response tokens generated by the model instead of max_tokens for controlling output length.. 

   * Linkup: Linkup is a web search service that is used in the evidence collection stage of the pipeline. The linkup_client.py module provides a client for interacting with the LinkUp API. This client is used to perform 
     web searches to gather information and evidence related to the claims extracted from the research brief. This allows the system to incorporate up-to-date, real-world information into the articles it generates.

   * Other Tools:
       * Flask: A lightweight web framework for Python, used to create the API endpoints.
       * Redis: Used as the message broker and result backend for Celery.
       * Pydantic: Used for data validation and settings management, ensuring that the data passed to the API and between different components of the system is well-formed.
