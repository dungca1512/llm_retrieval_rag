import unittest
import numpy as np
from unittest.mock import patch, MagicMock
from query import cosine_similarity, get_relevant_context, answer_question


class QueryTests(unittest.TestCase):

    def cosine_similarity_with_identical_vectors_returns_one(self):
        vec = np.array([1, 2, 3])
        self.assertEqual(cosine_similarity(vec, vec), 1)

    def cosine_similarity_with_orthogonal_vectors_returns_zero(self):
        vec1 = np.array([1, 0])
        vec2 = np.array([0, 1])
        self.assertAlmostEqual(cosine_similarity(vec1, vec2), 0)

    @patch('query.MongoClient')
    def get_relevant_context_returns_top_k_similar_documents(self, mock_client):
        mock_collection = MagicMock()
        mock_collection.find.return_value = [
            {'_id': 1, 'embedding': np.array([1, 2, 3]), 'file_path': 'path1'},
            {'_id': 2, 'embedding': np.array([4, 5, 6]), 'file_path': 'path2'},
            {'_id': 3, 'embedding': np.array([7, 8, 9]), 'file_path': 'path3'}
        ]
        mock_client.return_value.__getitem__.return_value.__getitem__.return_value = mock_collection
        _, metadatas = get_relevant_context("query", top_k=2)
        self.assertEqual(len(metadatas), 2)

    @patch('query.requests.post')
    def answer_question_returns_answer_from_api(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'candidates': [
                {'content': {'parts': [{'text': 'Answer from API'}]}}
            ]
        }
        mock_post.return_value = mock_response
        answer, _, _ = answer_question("query")
        self.assertEqual(answer, "Answer from API")

    @patch('query.requests.post')
    def answer_question_handles_api_error_gracefully(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {'error': {'message': 'API error'}}
        mock_post.return_value = mock_response
        answer, _, _ = answer_question("query")
        self.assertIn("Error: API error", answer)


if __name__ == '__main__':
    unittest.main()
