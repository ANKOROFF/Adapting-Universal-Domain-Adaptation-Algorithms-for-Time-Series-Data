from sklearn.metrics import accuracy_score

class BaseAlgorithm:
    def compute_h_score(self, src_true_labels, src_pred_labels, trg_true_labels, trg_pred_labels):
        """
        Вычисляет H-score для оценки производительности модели
        """
        # Вычисляем точность на исходном домене
        src_acc = accuracy_score(src_true_labels, src_pred_labels)
        
        # Вычисляем точность на целевом домене
        trg_acc = accuracy_score(trg_true_labels, trg_pred_labels)
        
        # Вычисляем H-score
        if src_acc + trg_acc == 0:
            return 0.0
            
        h_score = 2 * (src_acc * trg_acc) / (src_acc + trg_acc)
        return h_score

def get_backbone_class(backbone_name):
    """
    Возвращает класс backbone по его имени
    """
    if backbone_name == 'CNN':
        from models.backbones import CNN
        return CNN
    else:
        raise ValueError(f"Неизвестный backbone: {backbone_name}") 