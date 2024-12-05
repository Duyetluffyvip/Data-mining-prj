import weka.attributeSelection.CorrelationAttributeEval;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.InterquartileRange;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;
import weka.filters.unsupervised.instance.RemoveWithValues;

import java.io.File;

public class Preprocessing {
    public static void main(String[] args) throws Exception {
        // Set the base directory for your files
        String baseDir = "C:/Users/thedu/Desktop/Learn/Data Mining/weka/data/";

        /* Handle Missing Values */
        // Load dataset
        DataSource source = new DataSource(baseDir + "autism_screening.arff");
        Instances dataset = source.getDataSet();
        System.out.println("Loaded autism_screening.arff");

        // Replace missing values
        ReplaceMissingValues missingValues = new ReplaceMissingValues();
        missingValues.setInputFormat(dataset);
        Instances newData = Filter.useFilter(dataset, missingValues);
        System.out.println("Replaced missing values");

        // Save the dataset with missing values replaced
        ArffSaver saver = new ArffSaver();
        saver.setInstances(newData);
        saver.setFile(new File(baseDir + "missing.arff"));
        saver.writeBatch();
        System.out.println("Saved missing.arff");

        /* Detect Outliers and Extreme Values */
        // Load dataset with missing values replaced
        DataSource source1 = new DataSource(baseDir + "missing.arff");
        Instances dataset1 = source1.getDataSet();
        System.out.println("Loaded missing.arff");

        // Apply InterquartileRange filter
        String opt1[] = new String[]{"-R", "first-last", "-O", "3.0", "-E", "6.0"};
        InterquartileRange range = new InterquartileRange();
        range.setOptions(opt1);
        range.setInputFormat(dataset1);
        Instances rangedData = Filter.useFilter(dataset1, range);
        System.out.println("Applied InterquartileRange filter");

        // Save the dataset with outliers and extreme values detected
        saver.setInstances(rangedData);
        saver.setFile(new File(baseDir + "ranged.arff"));
        saver.writeBatch();
        System.out.println("Saved ranged.arff");

        /* Remove Outliers and Extreme Values */
        // Load dataset with detected outliers and extreme values
        DataSource source2 = new DataSource(baseDir + "ranged.arff");
        Instances dataset2 = source2.getDataSet();
        System.out.println("Loaded ranged.arff");

        String optOutlier[] = new String[]{"-S", "0.0", "-C", "22", "-L", "last"};
        String optExtreme[] = new String[]{"-S", "0.0", "-C", "23", "-L", "last"};

        // Remove outliers
        RemoveWithValues removeOutlier = new RemoveWithValues();
        removeOutlier.setOptions(optOutlier);
        removeOutlier.setInputFormat(dataset2);
        Instances removedOutlierData = Filter.useFilter(dataset2, removeOutlier);
        saver.setInstances(removedOutlierData);
        saver.setFile(new File(baseDir + "outlierRemoved.arff"));
        saver.writeBatch();
        System.out.println("Saved outlierRemoved.arff");

        // Remove extreme values
        DataSource source3 = new DataSource(baseDir + "outlierRemoved.arff");
        Instances dataset3 = source3.getDataSet();
        System.out.println("Loaded outlierRemoved.arff");

        RemoveWithValues removeExtremeValue = new RemoveWithValues();
        removeExtremeValue.setOptions(optExtreme);
        removeExtremeValue.setInputFormat(dataset3);
        Instances extreme = Filter.useFilter(dataset3, removeExtremeValue);
        saver.setInstances(extreme);
        saver.setFile(new File(baseDir + "extremeRemoved.arff"));
        saver.writeBatch();
        System.out.println("Saved extremeRemoved.arff");

        /* Remove Outlier and Extreme Value Attributes */
        source = new DataSource(baseDir + "extremeRemoved.arff");
        dataset = source.getDataSet();
        System.out.println("Loaded extremeRemoved.arff");

        String opt2[] = new String[]{"-R", "22,23"};
        Remove remove = new Remove();
        remove.setOptions(opt2);
        remove.setInputFormat(dataset);
        newData = Filter.useFilter(dataset, remove);
        System.out.println("Removed outlier and extreme value attributes");

        // Save the final dataset without outlier and extreme value attributes
        saver.setInstances(newData);
        saver.setFile(new File(baseDir + "final.arff"));
        saver.writeBatch();
        System.out.println("Saved final.arff");

        /* Correlation Analysis */
        source = new DataSource(baseDir + "final.arff");
        dataset = source.getDataSet();
        CorrelationAttributeEval cEval = new CorrelationAttributeEval();

        System.out.print("        ");
        for (int i = dataset.numAttributes() - 1; i >= 0; i--) {
            System.out.print("A[" + i + "] ");
        }
        System.out.println("");
        for (int i = dataset.numAttributes() - 1; i >= 0; i--) {
            dataset.setClassIndex(i);
            cEval.buildEvaluator(dataset);
            for (int j = 0; j <= i; j++) {
                System.out.print(String.format("%.2f", cEval.evaluateAttribute(j)) + "  ");
            }
            System.out.println("");
        }
        System.out.println("Correlation analysis completed");

        /* Discretize Attributes */
        source = new DataSource(baseDir + "final.arff");
        dataset = source.getDataSet();
        System.out.println("Loaded final.arff for discretization");

        // Apply Discretize filter
        String op3[] = new String[]{"-B", "10", "-M", "-1.0", "-R", "first-last"};
        Discretize discretize = new Discretize();
        discretize.setOptions(op3);
        discretize.setInputFormat(dataset);
        newData = Filter.useFilter(dataset, discretize);
        System.out.println("Applied Discretize filter");

        // Save the discretized dataset
        saver.setInstances(newData);
        saver.setFile(new File(baseDir + "discretized.arff"));
        saver.writeBatch();
        System.out.println("Saved discretized.arff");
    }
}
