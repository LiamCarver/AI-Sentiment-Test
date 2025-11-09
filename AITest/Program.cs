using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.Tokenizers;

class Program
{
    static void Main()
    {
        using var session = InitializeSession("distilbert_onnx/model.onnx");

        PrintModelMetadata(session);

        var exampleText = "I don't like this.";

        var tokenizer = InitializeTokenizer("distilbert_onnx/vocab.txt");
        var (inputIdsTensor, attentionMaskTensor) = PrepareInputs(tokenizer, exampleText);

        var results = RunInference(session, inputIdsTensor, attentionMaskTensor);
        InterpretResults(results);
    }

    private static InferenceSession InitializeSession(string modelPath)
    {
        var environment = OrtEnv.Instance();
        environment.EnvLogLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_VERBOSE;

        var options = new SessionOptions();
        options.AppendExecutionProvider_DML(); // Enable GPU
        options.EnableProfiling = true;

        return new InferenceSession(modelPath, options);
    }

    private static BertTokenizer InitializeTokenizer(string vocabPath)
    {
        return BertTokenizer.Create(vocabPath);
    }

    private static void PrintModelMetadata(InferenceSession session)
    {
        Console.WriteLine("Model Inputs:");
        foreach (var input in session.InputMetadata)
            Console.WriteLine($"  {input.Key} : {string.Join(", ", input.Value.Dimensions)} ({input.Value.ElementType})");

        Console.WriteLine("\nModel Outputs:");
        foreach (var output in session.OutputMetadata)
            Console.WriteLine($"  {output.Key}");
    }

    private static (DenseTensor<long> inputIds, DenseTensor<long> attentionMask)
        PrepareInputs(BertTokenizer tokenizer, string text)
    {
        var inputIds = tokenizer.EncodeToIds(text).Select(x => (long)x).ToArray();
        var attentionMask = inputIds.Select(id => id == tokenizer.PaddingTokenId ? 0L : 1L).ToArray();

        var inputIdsTensor = new DenseTensor<long>(inputIds, [1, inputIds.Length]);
        var attentionMaskTensor = new DenseTensor<long>(attentionMask, [1, attentionMask.Length]);

        return (inputIdsTensor, attentionMaskTensor);
    }

    private static IDisposableReadOnlyCollection<DisposableNamedOnnxValue> RunInference(
        InferenceSession session,
        DenseTensor<long> inputIds,
        DenseTensor<long> attentionMask)
    {
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input_ids", inputIds),
            NamedOnnxValue.CreateFromTensor("attention_mask", attentionMask)
        };

        return session.Run(inputs);
    }

    private static void InterpretResults(IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results)
    {
        var logits = results.First().AsTensor<float>().ToArray();

        float neg = logits[0];
        float pos = logits[1];

        float expNeg = MathF.Exp(neg);
        float expPos = MathF.Exp(pos);
        float sum = expNeg + expPos;

        float probNeg = expNeg / sum;
        float probPos = expPos / sum;

        Console.WriteLine($"Negative: {probNeg:P2}, Positive: {probPos:P2}");
        Console.WriteLine($"Predicted sentiment: {(probPos > probNeg ? "POSITIVE" : "NEGATIVE")}");
    }
}
