def test_application_modules_import():
    from nemotron_serving.jina_rerank import JinaReranker, _format_prompt
    from nemotron_serving.nemotron_rerank import NemotronReranker
    from nemotron_serving.server import NemotronScheduler, RerankScheduler
