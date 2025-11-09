using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.Tokenizers;

namespace AITest;

internal class BartOnnx
{
    public void Run()
    {
        using (var session = InitializeSession("bart_mnli_onnx/model.onnx"))
        {
            PrintModelMetadata(session);
            
            var exampleText = "I don't like this.";

            var tokenizer = InitializeTokenizer("bart_mnli_onnx/vocab.json", "bart_mnli_onnx/merges.txt");
        }
    }

    private InferenceSession InitializeSession(string modelPath)
    {
        var environment = OrtEnv.Instance();
        environment.EnvLogLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_VERBOSE;

        var options = new SessionOptions();
        options.AppendExecutionProvider_DML(); // Enable GPU
        options.EnableProfiling = true;

        return new InferenceSession(modelPath, options);
    }

    private BpeTokenizer InitializeTokenizer(string vocabPath, string mergesPath)
    {
        return BpeTokenizer.Create(vocabPath, mergesPath);
    }

    private void PrintModelMetadata(InferenceSession session)
    {
        Console.WriteLine("Model Inputs:");
        foreach (var input in session.InputMetadata)
            Console.WriteLine($"  {input.Key} : {string.Join(", ", input.Value.Dimensions)} ({input.Value.ElementType})");

        Console.WriteLine("\nModel Outputs:");
        foreach (var output in session.OutputMetadata)
            Console.WriteLine($"  {output.Key}");
    }
}
