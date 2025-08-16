using Microsoft.ML;
using SentimentApp;

var ml = new MLContext(seed: 1);

string dataPath =
    @"D:\PROJECTS\ML.NET\Data\sentiment+labelled+sentences\sentiment labelled sentences\data.txt";

// Load data (TSV, no header)
var data = ml.Data.LoadFromTextFile<InputData>(
    path: dataPath,
    hasHeader: false,
    separatorChar: '\t');

// Train/test split
var split = ml.Data.TrainTestSplit(data, testFraction: 0.2);

// Pipeline: Text -> TF-IDF features -> Logistic Regression
var pipeline =
    ml.Transforms.Text.FeaturizeText("Features", nameof(InputData.Text))
      .Append(ml.BinaryClassification.Trainers.SdcaLogisticRegression(
          labelColumnName: nameof(InputData.Label),
          featureColumnName: "Features"));

// Train
var model = pipeline.Fit(split.TrainSet);

// Evaluate
var preds = model.Transform(split.TestSet);
var metrics = ml.BinaryClassification.Evaluate(preds, labelColumnName: nameof(InputData.Label));

Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
Console.WriteLine($"AUC:      {metrics.AreaUnderRocCurve:P2}");
Console.WriteLine($"F1 Score: {metrics.F1Score:P2}");

// Save model (and reload to prove everything works)
const string modelPath = "sentimentModel.zip";
ml.Model.Save(model, split.TrainSet.Schema, modelPath);
Console.WriteLine($"Model saved to {modelPath}");

var loaded = ml.Model.Load(modelPath, out _);
var engine = ml.Model.CreatePredictionEngine<InputData, OutputPrediction>(loaded);

// Interactive test loop
Console.WriteLine();
while (true)
{
    Console.Write("Type a sentence (or 'exit'): ");
    var text = Console.ReadLine();
    if (string.Equals(text, "exit", StringComparison.OrdinalIgnoreCase)) break;

    var result = engine.Predict(new InputData { Text = text });
    Console.WriteLine($"Prediction: {(result.PredictedLabel ? "Positive ✅" : "Negative ❌")}");
    Console.WriteLine($"Prob(positive): {result.Probability:P2}");
    Console.WriteLine();
}
