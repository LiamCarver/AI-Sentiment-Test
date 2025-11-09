using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.Tokenizers;

var environment = OrtEnv.Instance();
environment.EnvLogLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_VERBOSE;

var options = new SessionOptions();
options.AppendExecutionProvider_DML(); // Enable GPU
options.EnableProfiling = true;

using (var session = new InferenceSession("distilbert_onnx/model.onnx", options))
{
    Console.WriteLine("Model Inputs:");
    foreach (var input in session.InputMetadata)
        Console.WriteLine($"  {input.Key} : {string.Join(", ", input.Value.Dimensions)} ({input.Value.ElementType})");

    Console.WriteLine("\nModel Outputs:");
    foreach (var output1 in session.OutputMetadata)
        Console.WriteLine($"  {output1.Key}");

    var exampleText = "I don't like this.";
    
    BertTokenizer tokenizer = BertTokenizer.Create("distilbert_onnx/vocab.txt");
    var inputIds = tokenizer.EncodeToIds(exampleText).Select(x => (long)x).ToArray();

    var attentionMask = inputIds.Select(id => id == tokenizer.PaddingTokenId ? 0L : 1L).ToArray();

    var inputIdsTensor = new DenseTensor<long>(inputIds, new[] { 1, inputIds.Length });
    var attentionMaskTensor = new DenseTensor<long>(attentionMask, new[] { 1, attentionMask.Length });

    var inputs = new List<NamedOnnxValue>
    {
        NamedOnnxValue.CreateFromTensor("input_ids", inputIdsTensor),
        NamedOnnxValue.CreateFromTensor("attention_mask", attentionMaskTensor)
    };

    using (IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = session.Run(inputs))
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