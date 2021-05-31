import com.mashape.unirest.http.Unirest;
import lombok.Getter;
import lombok.Setter;
import org.json.JSONObject;
import fr.inria.atlanmod.commons.log.Log;

import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;


/**
 * The main class to run a QuestionAnswering pipeline with Huggingface endpoints (language models).
 * <p>
 * To run this, it is needed to deploy a server that runs the language models to do HTTP requests to the server,
 * asking for answers to a given question and corpus.
 */
public class QuestionAnswering {

    /**
     * The address to the server running the language models.
     */
    @Getter
    @Setter
    private static String address = "http://127.0.0.1:5000/";


    public void testQA(String endpoint) {
        /*
        JSONObject request = new JSONObject();
        request.put("question", question);
        request.put("corpus", corpus);
        JSONObject response = new JSONObject();
        try {
            response = Unirest.post(address + endpoint)
                    .header("Content-Type", "application/json")
                    .body(request)
                    .asJson().getBody().getObject();
            this.setAnswer(response.getJSONObject("bert-large-uncased-whole-word-masking-finetuned-squad").getString("answer"));
            //System.out.println("| " + question + " | " +
                    //response.getJSONObject("bert-large-uncased-whole-word-masking-finetuned-squad").getString("answer") + " | ");
                    //"["+response.getJSONObject("bert-large-uncased-whole-word-masking-finetuned-squad").getInt("paragraph_id")+"] "+
                    //response.getJSONObject("bert-large-uncased-whole-word-masking-finetuned-squad").getString("answer") + " | ");
        } catch (Exception e) {
            Log.error(e, "An error occurred while computing the answer, see the attached exception");
        }

         */
    }

    /**
     * Process a QuestionAnswering request.
     *
     * @param corpus     the corpus
     * @param question   the question
     * @param endpoint   the endpoint where to do the request in {@link #address}
     * @param modelNames the name of the models that will compute an answer for the given question (i.e. one answer
     *                   per model)
     * @return the {@link List} containing all generated answers to the given question by the given language models.
     */
    public static List<AnswerObject> processQA(String corpus, String question, String endpoint,
                                               List<String> modelNames) {
        JSONObject request = new JSONObject();
        request.put("question", question);
        request.put("corpus", corpus);
        request.put("modelNames", modelNames);
        JSONObject response = new JSONObject();
        try {
            response = Unirest.post(address + endpoint)
                    .header("Content-Type", "application/json")
                    .body(request)
                    .asJson().getBody().getObject();
            List<AnswerObject> answerObjects = new ArrayList<>();
            for (String modelName : modelNames) {
                JSONObject modelResponse = response.getJSONObject(modelName);
                // TODO: check if null????
                String answer = modelResponse.getString("answer");
                int beginPosition = modelResponse.getInt("beginPosition");
                int endPosition = modelResponse.getInt("endPosition");
                AnswerObject answerObject = new AnswerObject(answer, modelName, beginPosition, endPosition);
                answerObjects.add(answerObject);
            }
            return answerObjects;
        } catch (Exception e) {
            Log.error(e, "An error occurred while computing the answer, see the attached exception");
        }
        return null;
    }

    /**
     * Sets the given language models in the remote server so they are ready to use.
     *
     * @param modelNames the model names
     * @param endpoint   the endpoint where to do the request in {@link #address}
     */
    public static void setModelNames(List<String> modelNames, String endpoint) {
        JSONObject request = new JSONObject();
        request.put("modelNames", modelNames);
        try {
            JSONObject response = new JSONObject();
            response = Unirest.post(address + endpoint)
                    .header("Content-Type", "application/json")
                    .body(request)
                    .asJson().getBody().getObject();
            // TODO: check status ok
        } catch (Exception e) {
            Log.error(e, "An error occurred while computing the answer, see the attached exception");
        }
    }

    /**
     * The entry point of application.
     * <p>
     * It allows to set the desired language models that will perform the QuestionAnswering, and then to write
     * questions that will be answered using the given corpus file as the place where the answers should be found.
     * <p>
     * If the input is "print" (instead of a question), then all computed Question-Answer pairs are printed
     * @param args the input arguments.
     */
    public static void main(String[] args) {
        // The corpus file must be in the resources folder (/src/main/resources/)
        QAObject qaObject = new QAObject("pablo_picasso.txt");
        List<String> modelNames = new ArrayList<String>();
        modelNames.add("ktrapeznikov/albert-xlarge-v2-squad-v2");
        modelNames.add("twmkn9/albert-base-v2-squad2");
        //modelNames.add("mrm8488/bert-tiny-5-finetuned-squadv2");
        //modelNames.add("bert-large-uncased-whole-word-masking-finetuned-squad");
        //modelNames.add("distilbert-base-cased-distilled-squad");
        //modelNames.add("valhalla/longformer-base-4096-finetuned-squadv1");
        //modelNames.add("google/bigbird-base-trivia-itc");

        //setModelNames(modelNames, "set-models");

        Scanner in = new Scanner(System.in);
        while (true) {
            System.out.print("Question: ");
            String question = in.nextLine();
            if (question.equals("print")) {
                qaObject.printQAPairs();
            }
            else {
                List<AnswerObject> answerObjects = processQA(qaObject.getCorpus(), question, "transformer-qa", modelNames);
                qaObject.addQAPair(question, answerObjects);
                System.out.println("Answers have been stored. Enter \"print\" to see all question-answers");
            }
        }
    }

}
