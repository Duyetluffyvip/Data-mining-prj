import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;

import java.io.File;

public class CSVtoArff {
    public static void main(String[] args) throws Exception {

        // Specify the file path
        File file = new File("C:/Users/thedu/Desktop/Learn/Data Mining/weka/data/autism_screening.csv");

        // Check if the file exists
        if (!file.exists()) {
            System.out.println("File not found!");
            return;
        }

        // Load CSV
        CSVLoader loader = new CSVLoader();
        loader.setSource(file);
        Instances data = loader.getDataSet(); // Get instances object

        // Save ARFF
        ArffSaver saver = new ArffSaver();
        saver.setInstances(data); // Set the dataset we want to convert
        saver.setFile(new File("C:/Users/thedu/Desktop/Learn/Data Mining/weka/data/autism_screening.arff"));
        saver.writeBatch(); // Write to ARFF

    }
}
