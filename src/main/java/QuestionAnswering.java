import com.mashape.unirest.http.Unirest;
import lombok.Getter;
import lombok.Setter;
import org.json.JSONObject;
import fr.inria.atlanmod.commons.log.Log;
import org.apache.commons.io.IOUtils;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;

import static java.util.Objects.isNull;

public class QuestionAnswering {

    @Getter
    @Setter
    private String question;

    @Getter
    private String answer;

    @Getter
    @Setter
    private String corpus;

    @Getter
    private int beginPosition;

    @Getter
    private int endPosition;

    @Getter
    @Setter
    private String address;

    private final String CORPUS_FILE = "corpus.txt";

    public void requestAnswer() {
        JSONObject request = new JSONObject();
        request.put("question", question);
        request.put("corpus", corpus);
        JSONObject response = new JSONObject();
        try {
            response = Unirest.post(address + "qa")
                    .header("Content-Type", "application/json")
                    .body(request)
                    .asJson().getBody().getObject();
            answer = response.getString("answer");
            beginPosition = response.getInt("beginPosition");
            endPosition = response.getInt("endPosition");
        } catch (Exception e) {
            Log.error(e, "An error occurred while computing the answer, see the attached exception");
        }
    }

    public QuestionAnswering() {
        try (InputStream inputStream =
                     this.getClass().getResourceAsStream(CORPUS_FILE)) {
            if (isNull(inputStream)) {
                Log.error("Cannot find the file {0}, this processor won't get any corpus", CORPUS_FILE);
            } else {
                setCorpus(IOUtils.toString(inputStream, StandardCharsets.UTF_8));
            }
        } catch (IOException e) {
            Log.error("An error occurred when processing the corpus file {0}, this processor may produce "
                    + "unexpected behavior. Check the logs for more information.", CORPUS_FILE);
        }
        Log.info("Loaded corpus from {0}", CORPUS_FILE);
    }

    public static void main(String[] args) {
        QuestionAnswering qa = new QuestionAnswering();
        qa.setQuestion("<Question goes here>");
        qa.setAddress("http://127.0.0.1:5000/");
        qa.requestAnswer();
        System.out.println("Question: " + qa.getQuestion());
        System.out.println("Answer: " + qa.getAnswer());
    }

}
