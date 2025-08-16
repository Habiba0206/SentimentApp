using Microsoft.ML.Data;

namespace SentimentApp;

public sealed class OutputPrediction
{
    [ColumnName("PredictedLabel")]
    public bool PredictedLabel { get; set; }

    public float Probability { get; set; }
    public float Score { get; set; }
}