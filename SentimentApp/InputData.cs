using Microsoft.ML.Data;

namespace SentimentApp;

public sealed class InputData
{
    /// <summary>
    /// Column 0: the sentence text.
    /// </summary>
    [LoadColumn(0)]
    public string Text { get; set; }

    /// <summary>
    /// Column 1: the label (0/1). If your file uses 0/1, keep bool here.
    /// </summary>
    [LoadColumn(1)]
    public bool Label { get; set; }
}