SEPARATORS = {'。', '.', '！', '!', '？', '?', '…', '/n'}


class SentenceSeg:
    def __init__(self):
        pass

    @staticmethod
    def regex_seg(content):
        """
        正则断句
        :param content: A string of Chinese text.
        :return: A list of strings, where each string is a sentence.
        """

        sentences = []
        start = 0
        for i, char in enumerate(content):
            if char in SEPARATORS:
                sentences.append(content[start:i + 1])
                start = i + 1

        # Append the last sentence if it's not empty.
        if start < len(content):
            sentences.append(content[start:])

        return sentences

    def algo_seg(self, content):
        """
        算法断句叠加正则断句
        :param content:
        :return:
        """
        pass


if __name__ == '__main__':
    seg = SentenceSeg()

    text = """
    前排空间： Model 3的前排空间非常宽敞，驾驶座椅的舒适度极高，可以电动调节，找到最适合自己的驾驶姿势。
    副驾驶座椅同样舒适，而且可以前后移动，为乘客提供更大的腿部空间。中控台和方向盘的设计简洁而实用，操作便捷。
    特别是方向盘的尺寸适中，手感极佳，让我在驾驶时感到非常自信。 二排空间： Model 3的二排空间相对宽敞，两个座椅之间的距离适中，乘坐舒适。
    然而，由于车辆的设计原因，后排乘客的头部空间可能会感到略显局促。但是，对于一般的乘客来说，这个问题并不明显。
    此外，二排座椅可以按比例放倒，以增加后备箱的储物空间。 后备箱储物空间： 后备箱储物空间深度和宽度都足够放置大型行李箱和其他杂物。
    此外，后备箱内部设计规整，方便整理和摆放物品
    """

    print(seg.regex_seg(text))
