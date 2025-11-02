import src.convert_labels as convert_labels
import src.ml_build_dataset as ml_build_dataset
import src.ml_train as ml_train
import src.ml_predict as ml_predict

def main():
    
    print("Converting labels")
    convert_labels.parse_excel(
        input_xlsx="devs_similarity_t=0.65.xlsx",
        output_labels_csv="labels_from_excel.csv",
        output_candidates_csv="candidates_from_excel.csv"
        )


    print("Building training dataset")
    ml_build_dataset.build_dataset(candidates_csv="candidates_from_excel.csv",
                                   labels_csv="labels_from_excel.csv",
                                   out_csv="train_dataset.csv"
                                   )

    print("Training logistic regression model")
    ml_train.train_and_eval(train_csv="train_dataset.csv",model_out="logreg.pkl")

    print("Scoring candidate pairs with trained model")
    ml_predict.score_candidates(candidates_csv="devs_similarity.csv",
                                model_pkl="logreg.pkl",
                                out_csv="3ml_scored_p0915.csv",
                                threshold=0.65
                                )

if __name__ == "__main__":
    main()
