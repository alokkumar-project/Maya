from ml_model.Transformer import Transformer
class ChatEngine:

    def __init__(self, token_file="all_token.pkl", model_file="maya_v1.pkl"):
        self._model = Transformer(token_file=token_file, training_mode='n',encoder_unit=3,decoder_unit=3,embedding_dim=128,multi_head_decoder=4,multi_head_encoder=4)
        self._model.load_model(file_name=model_file)

    def respond(self, message: str, mode: str = "beam") -> str:
        """
        mode: "beam"   -> self._model.generate()          (beam search)
              "greedy" -> self._model.gready_generate()    (greedy decoding)
        """
        message = (message or "").strip()
        if not message:
            return "Please type a message."
        try:
            if mode == "greedy":
                return self._model.gready_generate(message)
            return self._model.generate(message)
        except Exception as exc:
            print("Generation error:", exc)
            return "Sorry, I couldn't generate a response for that just now."


_engine_instance = None


def load_chatbot(token_file="token_file.pkl", model_file="model.pkl"):
    """Loads the model once and reuses it for every request (singleton)."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = ChatEngine(token_file=token_file, model_file=model_file)
    return _engine_instance
